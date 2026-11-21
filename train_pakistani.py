"""Download a few Wikipedia food images and train the 4-dish classifier head.

Run once with:  python train_pakistani.py
It creates data/pakistani_images/ and pakistani_head.pt.
"""

from io import BytesIO
from pathlib import Path
import re

import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "pakistani_images"
HEAD_FILE = BASE / "pakistani_head.pt"

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "NutriVisionStudentProject/1.0 (educational use)"}
PER_CLASS = 25
EPOCHS = 10

QUERIES = {
    "Biryani": ["biryani", "biriyani"],
    "Samosa": ["samosa", "samosas"],
    "Nihari": ["nihari"],
    "Gulab Jamun": ["gulab jamun"],
}

MATCH = {
    "Biryani": lambda t: bool(re.search(r"\b(biryani|biriyani)\b", t)),
    "Samosa": lambda t: bool(re.search(r"\bsamosas?\b", t)),
    "Nihari": lambda t: bool(re.search(r"\bnihari\b", t)),
    "Gulab Jamun": lambda t: bool(re.search(r"\bgulab\b.*\bjamun\b|\bgulabjamun\b", t)),
}


def api(params):
    response = requests.get(API, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def search_titles(query, limit=80):
    data = api(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 6,
            "srlimit": limit,
            "format": "json",
        }
    )
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def get_thumb_urls(titles):
    urls = []
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        data = api(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 320,
                "format": "json",
            }
        )
        for page in data.get("query", {}).get("pages", {}).values():
            info = page.get("imageinfo", [{}])[0]
            if info.get("thumburl"):
                urls.append(info["thumburl"])
    return urls


def download_images():
    for food, queries in QUERIES.items():
        folder = DATA_DIR / food
        folder.mkdir(parents=True, exist_ok=True)
        if len(list(folder.glob("*"))) >= PER_CLASS:
            print(f"{food}: already has {len(list(folder.glob('*')))} images")
            continue

        titles = []
        for query in queries:
            titles += search_titles(query)

        titles = list(dict.fromkeys(t for t in titles if MATCH[food](t.lower())))
        saved = len(list(folder.glob("*")))

        for url in get_thumb_urls(titles):
            if saved >= PER_CLASS:
                break
            try:
                response = requests.get(url, headers=HEADERS, timeout=20)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(folder / f"{saved:03d}.jpg", quality=90)
                saved += 1
            except Exception:
                continue

        print(f"{food}: downloaded {saved} images")


def train():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    plain_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    plain = datasets.ImageFolder(DATA_DIR, transform=plain_transform)
    classes = full.classes
    print("Classes:", classes)

    indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(42)).tolist()
    val_size = max(1, int(0.2 * len(indices)))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_loader = DataLoader(
        Subset(full, train_idx), batch_size=16, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        Subset(plain, val_idx), batch_size=16, shuffle=False, num_workers=0
    )

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, len(classes)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / max(total, 1)
        best_accuracy = max(best_accuracy, accuracy)
        print(f"Epoch {epoch + 1:02d} | loss {total_loss:.3f} | val acc {accuracy:.2%}")

    torch.save(
        {
            "classes": classes,
            "state_dict": model.classifier.state_dict(),
            "val_accuracy": best_accuracy,
        },
        HEAD_FILE,
    )
    print(f"Saved {HEAD_FILE} (best val accuracy {best_accuracy:.2%})")


if __name__ == "__main__":
    download_images()
    train()
