"""NutriVision - simple single-file app.

Flow: upload image -> predict food -> choose grams -> save meal -> view charts.
"""

from datetime import date, datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

BASE = Path(__file__).resolve().parent
NUTRITION_FILE = BASE / "data" / "nutrition.csv"
DIARY_FILE = BASE / "data" / "diary.csv"
LOCAL_FILE = BASE / "pakistani_head.pt"

NUTRIENTS = ["Calories", "Protein", "Carbs", "Fats"]
COLUMNS = ["Date", "Time", "Food", "Grams"] + NUTRIENTS
LOCAL_THRESHOLD = 0.45

# ImageNet label (lower-case) -> name used in nutrition.csv.
FOOD_NAMES = {
    "trifle": "Trifle",
    "ice cream": "Ice cream",
    "ice lolly": "Ice lolly",
    "french loaf": "French loaf",
    "bagel": "Bagel",
    "pretzel": "Pretzel",
    "cheeseburger": "Cheeseburger",
    "hotdog": "Hotdog",
    "mashed potato": "Mashed potato",
    "granny smith": "Apple",
    "strawberry": "Strawberry",
    "orange": "Orange",
    "lemon": "Lemon",
    "fig": "Fig",
    "pineapple": "Pineapple",
    "banana": "Banana",
    "jackfruit": "Jackfruit",
    "custard apple": "Custard apple",
    "pomegranate": "Pomegranate",
    "carbonara": "Carbonara",
    "chocolate sauce": "Chocolate sauce",
    "meat loaf": "Meat loaf",
    "pizza": "Pizza",
    "potpie": "Potpie",
    "burrito": "Burrito",
    "espresso": "Espresso",
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------- Data ----------------

def nutrition():
    return pd.read_csv(NUTRITION_FILE).set_index("Food").to_dict("index")


def calc(food, grams, table):
    row = table[food]
    factor = grams / 100
    return {
        "Food": food,
        "Grams": float(grams),
        **{key: round(float(row[key]) * factor, 1) for key in NUTRIENTS},
    }


def read_diary(path=DIARY_FILE):
    if not path.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
    return pd.read_csv(path)


def add_meal(meal, path=DIARY_FILE):
    now = datetime.now()
    row = {"Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M"), **meal}
    data = pd.concat([read_diary(path), pd.DataFrame([row])], ignore_index=True)
    data.to_csv(path, index=False)


def delete_meal(index, path=DIARY_FILE):
    read_diary(path).drop(index).reset_index(drop=True).to_csv(path, index=False)


def totals(day, path=DIARY_FILE):
    data = read_diary(path)
    data = data[data["Date"] == day]
    if data.empty:
        return {key: 0.0 for key in NUTRIENTS}
    return {key: float(data[key].sum()) for key in NUTRIENTS}


def week(day, path=DIARY_FILE):
    end = pd.to_datetime(day)
    days = pd.date_range(end - pd.Timedelta(days=6), end).strftime("%Y-%m-%d")

    data = read_diary(path)
    data = data[data["Date"].isin(days)]
    data = data.groupby("Date")[NUTRIENTS].sum()
    data = data.reindex(days, fill_value=0).round(1).rename_axis("Date")
    return data.reset_index()

# ---------------- Models ----------------

def load_general():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    return model, weights.meta["categories"]


def load_local():
    """Return the small Pakistani-dish model, or None if not trained."""
    if not LOCAL_FILE.exists():
        return None

    saved = torch.load(LOCAL_FILE, map_location="cpu", weights_only=False)
    classes = saved["classes"]
    model, _ = load_general()
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, len(classes)))
    model.classifier.load_state_dict(saved["state_dict"])
    model.eval()
    return model, classes


@lru_cache(maxsize=1)
def models():
    return load_general(), load_local()


def run_model(model, image, classes):
    x = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(x)[0], dim=0)
    confidence, index = probabilities.topk(1)
    return classes[int(index)], float(confidence)


def predict(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    (general_model, general_classes), local = models()

    # Rule 1: if the local dish model is confident, use it.
    if local:
        label, confidence = run_model(local[0], image, local[1])
        if confidence >= LOCAL_THRESHOLD:
            return label, confidence, "Pakistani dish model"

    # Rule 2: otherwise use the general ImageNet model.
    label, confidence = run_model(general_model, image, general_classes)
    food = FOOD_NAMES.get(label.lower(), label)
    return food, confidence, "ImageNet model"

# ---------------- Checks ----------------

def run_checks():
    lines = ["# NutriVision evaluation", ""]
    table = nutrition()

    try:
        expected = round(float(table["Pizza"]["Calories"]) * 2, 1)
        assert calc("Pizza", 200, table)["Calories"] == expected
        lines.append("- PASS: Nutrition calculation")
    except Exception as error:
        lines.append(f"- FAIL: Nutrition calculation - {error}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diary.csv"
            add_meal(calc("Pizza", 100, table), path)
            assert len(read_diary(path)) == 1
            delete_meal(0, path)
            assert read_diary(path).empty
        lines.append("- PASS: Diary save / delete")
    except Exception as error:
        lines.append(f"- FAIL: Diary storage - {error}")

    try:
        assert len(load_general()[1]) == 1000
        lines.append("- PASS: ImageNet model")
    except Exception as error:
        lines.append(f"- FAIL: ImageNet model - {error}")

    try:
        local = load_local()
        assert local and len(local[1]) == 4
        lines.append("- PASS: Pakistani dish model")
    except Exception as error:
        lines.append(f"- FAIL: Pakistani dish model - {error}")

    return "\n".join(lines) + "\n"

# ---------------- UI ----------------

def main():
    st.set_page_config(page_title="NutriVision", page_icon="🥗", layout="wide")
    st.title("NutriVision")
    st.caption("Upload food, estimate nutrition, save meals, see charts.")

    table = nutrition()

    with st.sidebar.expander("⚙️ Health checks"):
        if st.button("Run checks"):
            report = run_checks()
            st.code(report)
            st.download_button("Download report", report, file_name="evaluation_report.md")

    recognize, diary_tab, dashboard = st.tabs(["🍽️ Recognize", "📔 Diary", "📊 Dashboard"])

    with recognize:
        st.header("Recognize food")
        uploaded = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"])

        if uploaded is None:
            st.info("Upload one food item to begin.")
        else:
            image_bytes = uploaded.getvalue()
            left, right = st.columns([1, 1])
            left.image(image_bytes, caption="Uploaded image", use_container_width=True)

            with right:
                with st.spinner("Recognizing..."):
                    food, confidence, source = predict(image_bytes)

                st.success(f"**{food}**")
                st.write(f"Confidence: {confidence:.0%}")
                st.caption(f"Model: {source}")

                names = sorted(table)
                default = names.index(food) if food in table else 0
                chosen = st.selectbox("Food (edit if wrong)", names, index=default)
                grams = st.number_input("Grams", 1, 2000, 100, 10)

                meal = calc(chosen, grams, table)
                cols = st.columns(4)
                for col, key in zip(cols, NUTRIENTS):
                    col.metric(key, meal[key])

                if st.button("Add to diary", type="primary"):
                    add_meal(meal)
                    st.success(f"Saved {chosen} ({grams} g).")

    with diary_tab:
        st.header("Food diary")
        data = read_diary()
        if data.empty:
            st.info("No meals saved yet.")
        else:
            st.dataframe(data, use_container_width=True)
            row = st.number_input("Row to delete", 0, len(data) - 1, 0)
            if st.button("Delete row"):
                delete_meal(int(row))
                st.rerun()

    with dashboard:
        st.header("Nutrition dashboard")
        day = st.date_input("Date", value=date.today()).isoformat()

        values = totals(day)
        cols = st.columns(4)
        for col, key in zip(cols, NUTRIENTS):
            col.metric(key, values[key])

        st.subheader("Last 7 days")
        st.bar_chart(week(day).set_index("Date"))


if __name__ == "__main__":
    main()
