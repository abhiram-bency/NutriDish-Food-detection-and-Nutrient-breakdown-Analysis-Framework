# 🍽️ NutriDish — Computer Vision Nutrient Analysis System

> Food image → AI classification → Nutritional breakdown → REST API

Built with **MobileNetV2 + TensorFlow**, **Food-101 (streamed)**, and **FastAPI**.

---

## 📁 Project Structure

```
nutridish/
├── model/
│   ├── train.py          # ← ML training pipeline (streaming + transfer learning)
│   └── inference.py      # ← Prediction engine
├── api/
│   ├── main.py           # ← FastAPI application
│   ├── database.py       # ← SQLite ORM layer
│   └── schemas.py        # ← Pydantic request/response models
├── data/
│   └── nutrition_db.py   # ← Nutritional values for all 101 food classes
├── scripts/
│   └── test_pipeline.py  # ← End-to-end test script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd nutridish
python model/train.py
```

This will:
- Stream Food-101 from HuggingFace (no full download)
- Train MobileNetV2 in two phases (head → fine-tune)
- Save `nutridish_model.keras`

> ⏱️ **Time estimate**: ~1–2 h on GPU, ~4–8 h on CPU.  
> To cut time during development, reduce `STEPS_PER_EPOCH` in `train.py`.

### 3. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

### 4. Test the pipeline

```bash
# Basic test (no model required for nutrition DB check)
python scripts/test_pipeline.py

# With inference test
python scripts/test_pipeline.py --image path/to/burger.jpg
```

---

## 🌐 API Endpoints

### `POST /analyze-image`
Upload a food photo → get predicted food + nutritional values.

```bash
curl -X POST "http://localhost:8000/analyze-image" \
  -F "file=@burger.jpg"
```

**Response:**
```json
{
  "food": "hamburger",
  "confidence": 0.9312,
  "calories": 295,
  "protein": 17,
  "carbs": 30,
  "fat": 14,
  "top_predictions": [
    {"label": "hamburger", "confidence": 0.9312},
    {"label": "hot_dog",   "confidence": 0.0312},
    {"label": "club_sandwich", "confidence": 0.0198}
  ]
}
```

---

### `POST /log-food`
Store a food entry in the daily tracker.

```bash
curl -X POST "http://localhost:8000/log-food" \
  -H "Content-Type: application/json" \
  -d '{"food":"hamburger","calories":295,"protein":17,"carbs":30,"fat":14}'
```

---

### `GET /daily-summary`
Get total nutrient intake for a day.

```bash
# Today
curl "http://localhost:8000/daily-summary"

# Specific date
curl "http://localhost:8000/daily-summary?target_date=2025-07-15"
```

**Response:**
```json
{
  "date": "2025-07-15",
  "meals": 3,
  "total_calories": 712.0,
  "total_protein": 42.0,
  "total_carbs": 89.0,
  "total_fat": 28.0
}
```

---

### `GET /logs`
List all food entries for a day.

```bash
curl "http://localhost:8000/logs?target_date=2025-07-15"
```

---

### `DELETE /logs/{id}`
Remove a specific entry.

```bash
curl -X DELETE "http://localhost:8000/logs/3"
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
  └── MobileNetV2 backbone (ImageNet weights, frozen in Phase 1)
        └── GlobalAveragePooling2D
              └── BatchNorm → Dense(512, relu) → Dropout(0.4)
                    └── Dense(256, relu) → Dropout(0.3)
                          └── Dense(101, softmax)   ← 101 Food-101 classes
```

**Training Strategy:**

| Phase | What trains | LR | Epochs |
|-------|------------|-----|--------|
| 1 | Classification head only | 1e-3 | 10 |
| 2 | Head + top MobileNetV2 layers | 1e-5 | 20 |

---

## 📊 Dataset

[Food-101](https://huggingface.co/datasets/food101) via HuggingFace Datasets.

- 101 food categories
- 75,750 training / 25,250 validation images
- Loaded in **streaming mode** — no full download required

```python
from datasets import load_dataset
dataset = load_dataset("food101", split="train", streaming=True)
```

---

## 🗄️ Database Schema

**Table: `food_logs`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| food | TEXT | Predicted food label |
| confidence | REAL | Model confidence (0–1) |
| calories | REAL | kcal per 100g |
| protein | REAL | g per 100g |
| carbs | REAL | g per 100g |
| fat | REAL | g per 100g |
| log_date | DATE | Day of the meal |
| logged_at | DATETIME | Exact timestamp |
| notes | TEXT | Optional notes |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUTRIDISH_MODEL` | `nutridish_model.keras` | Path to saved model |
| `NUTRIDISH_DB` | `nutridish.db` | SQLite database path |

---

## 📱 Android Integration

The API is CORS-enabled. From Kotlin (Retrofit):

```kotlin
interface NutriDishApi {
    @Multipart
    @POST("analyze-image")
    suspend fun analyzeImage(@Part image: MultipartBody.Part): AnalyzeResponse

    @POST("log-food")
    suspend fun logFood(@Body body: LogFoodRequest): LogFoodResponse

    @GET("daily-summary")
    suspend fun getDailySummary(): DailySummaryResponse
}
```

---

## 📝 Notes

- All nutritional values are **per 100g serving** (approximate)
- If a food is not in the nutrition DB, default values are returned
- The model file (`nutridish_model.keras`) is **not included** — train it first

## Activate Enviornment
on bash:-
source .venv/bin/activate

## UI

To run UI :-
on bash :- 
uvicorn api.main:app --reload --port 8000

## Access UI on Browser
browser - http://127.0.0.1:8000/ui