# NutriVision

NutriVision is a small nutrition tracker I built for my PAI project. You upload a photo of your food, the app guesses what it is, lets you correct the guess, and saves the meal with calories and macros. It also draws a simple 7-day chart so you can see your week at a glance.

It runs locally with Streamlit. The first prediction downloads the pretrained model weights; after that, the app works without the internet.

## What it does

- Recognizes common foods like pizza, burgers, bananas, and more
- Also recognizes **Biryani, Samosa, Nihari, and Gulab Jamun** with a tiny extra model
- Lets you change the food name if the guess is wrong
- Lets you set the portion size in grams
- Estimates calories, protein, carbs, and fats
- Saves meals to a CSV food diary
- Shows today's totals and a 7-day chart
- Runs quick health checks from the sidebar or command line

## Run it

```bash
cd NutriVision
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

The first image may take a few seconds because the model has to load. Later images are much faster.

## How to use it

1. Open the **Recognize** tab and upload a JPG or PNG.
2. Check the predicted food. If it is wrong, pick the right one from the dropdown.
3. Enter the weight in grams and press **Add to diary**.
4. Open the **Diary** tab to see saved meals or delete a row.
5. Open the **Dashboard** tab for daily totals and the last 7 days.

There is also a small **Health checks** panel in the sidebar. Click **Run checks** if you want to make sure everything is working.

## What is in the folder

| File | What it is for |
| --- | --- |
| `app.py` | The whole app: recognition, nutrition math, diary, checks, and UI |
| `evaluate.py` | Runs the checks from the command line |
| `train_pakistani.py` | Optional trainer for the four local dishes |
| `pakistani_head.pt` | Trained 4-dish classifier |
| `data/nutrition.csv` | Nutrition values per 100 g |
| `data/diary.csv` | Your saved meals, created automatically |
| `data/pakistani_images/` | Training images, downloaded only if you retrain |

`app.py` is split into four easy sections: **Data**, **Models**, **Checks**, and **UI**.

## How recognition works

The app uses two models:

1. **MobileNetV2** handles general ImageNet foods.
2. A small **Pakistani dish model** handles Biryani, Samosa, Nihari, and Gulab Jamun.

If the local dish model is confident enough (45% or more), its answer wins. Otherwise, the app falls back to MobileNetV2. The UI tells you which model was used so you are not left guessing.

## Retrain the local dishes

The four local dishes were trained on freely licensed Wikipedia images. If you want to rebuild the small model:

```bash
python train_pakistani.py
```

This downloads images into `data/pakistani_images/` and updates `pakistani_head.pt`. Those images are ignored by git so the repository stays small.

## Health checks

```bash
python evaluate.py
```

This writes a short `evaluation_report.md` showing whether the nutrition math, diary storage, and both models are working.

## A few honest notes

- Nutrition values are rough estimates, not lab measurements.
- The model can be wrong, so the manual correction box is there for a reason.
- The local dish model was trained on a small dataset, so treat it as a learning project, not a medical tool.
- The external nutrition API from the proposal was intentionally skipped to keep the app simple and offline.

Built as a simple, hands-on project to show how image recognition and everyday nutrition tracking can fit together.
