"""
NutriDish — FastAPI Backend

Run:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import io
import sys
import os
from datetime import date
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# ─── Internal imports ─────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.database import init_db, get_db, create_food_log, get_daily_logs, get_daily_summary
from api.schemas import (
    AnalyzeImageResponse,
    LogFoodRequest,
    LogFoodResponse,
    DailySummaryResponse,
    LogsResponse,
    PredictionResult,
)
from data.nutrition_db import get_nutrition
from model.inference import FoodPredictor

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NutriDish API",
    description="Computer Vision Nutrient Analysis System — Food-101 + MobileNetV2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model singleton ──────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("NUTRIDISH_MODEL", "nutridish_model.keras")
_predictor: Optional[FoodPredictor] = None


def get_predictor() -> FoodPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FoodPredictor(MODEL_PATH)
    return _predictor


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    print("✅ Database initialised")
    # Warm up model
    try:
        get_predictor()
        print("✅ Model loaded")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   POST /analyze-image will return 503 until the model is trained.")


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "NutriDish API", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health():
    model_ready = _predictor is not None
    return {"status": "ok", "model_loaded": model_ready}


# ─── 1. POST /analyze-image ───────────────────────────────────────────────────

@app.post(
    "/analyze-image",
    response_model=AnalyzeImageResponse,
    summary="Upload a food photo and get nutritional breakdown",
    tags=["analysis"],
)
async def analyze_image(
    file: UploadFile = File(..., description="JPEG / PNG food image"),
):
    """
    Upload a food image and receive:
    - Predicted food label
    - Confidence score
    - Calorie + macro breakdown (per 100 g)
    """
    predictor = get_predictor()  # Raises 503-ready error if not loaded

    # ── Read image bytes ──────────────────────────────────────────────────────
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG / PNG).")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:   # 10 MB guard
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")

    try:
        from PIL import Image
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = predictor.predict(pil_image, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # ── Nutrition lookup ─────────────────────────────────────────────────────
    nutrition = get_nutrition(result["label"])

    return AnalyzeImageResponse(
        food=result["label"],
        confidence=round(result["confidence"], 4),
        calories=nutrition["calories"],
        protein=nutrition["protein"],
        carbs=nutrition["carbs"],
        fat=nutrition["fat"],
        top_predictions=[
            PredictionResult(label=p["label"], confidence=round(p["confidence"], 4))
            for p in result["top_k"]
        ],
    )


# ─── 2. POST /log-food ────────────────────────────────────────────────────────

@app.post(
    "/log-food",
    response_model=LogFoodResponse,
    status_code=201,
    summary="Log a meal to daily tracker",
    tags=["tracking"],
)
async def log_food(
    body: LogFoodRequest,
    db: Session = Depends(get_db),
):
    """
    Store a food entry (from /analyze-image or manual) in the daily log.
    """
    entry = create_food_log(
        db=db,
        food=body.food,
        confidence=body.confidence,
        calories=body.calories,
        protein=body.protein,
        carbs=body.carbs,
        fat=body.fat,
        notes=body.notes,
        log_date=body.log_date,
    )
    return entry


# ─── 3. GET /daily-summary ────────────────────────────────────────────────────

@app.get(
    "/daily-summary",
    response_model=DailySummaryResponse,
    summary="Total nutrient intake for a day",
    tags=["tracking"],
)
async def daily_summary(
    target_date: Optional[date] = Query(
        None, description="ISO date (YYYY-MM-DD). Defaults to today."
    ),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated calories and macros for the given date (today if omitted).
    """
    summary = get_daily_summary(db, target_date)
    return summary


# ─── 4. GET /logs ─────────────────────────────────────────────────────────────

@app.get(
    "/logs",
    response_model=LogsResponse,
    summary="All food entries for a day",
    tags=["tracking"],
)
async def get_logs(
    target_date: Optional[date] = Query(
        None, description="ISO date (YYYY-MM-DD). Defaults to today."
    ),
    db: Session = Depends(get_db),
):
    """List every logged food entry for the given date."""
    target = target_date or date.today()
    logs = get_daily_logs(db, target)
    return LogsResponse(
        date=str(target),
        count=len(logs),
        logs=logs,
    )


# ─── 5. DELETE /logs/{log_id} ─────────────────────────────────────────────────

@app.delete(
    "/logs/{log_id}",
    status_code=204,
    summary="Delete a food log entry",
    tags=["tracking"],
)
async def delete_log(log_id: int, db: Session = Depends(get_db)):
    from api.database import FoodLog
    entry = db.query(FoodLog).filter(FoodLog.id == log_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found.")
    db.delete(entry)
    db.commit()


# ─── Serve UI ─────────────────────────────────────────────────────────────────

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib

UI_DIR = pathlib.Path(__file__).parent.parent / "ui"

@app.get("/ui", include_in_schema=False)
async def serve_ui():
    return FileResponse(UI_DIR / "index.html")