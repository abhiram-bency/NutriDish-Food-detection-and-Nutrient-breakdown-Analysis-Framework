"""
NutriDish — Pydantic Schemas
"""

from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── Nutrition ────────────────────────────────────────────────────────────────

class NutritionInfo(BaseModel):
    calories: float
    protein:  float
    carbs:    float
    fat:      float


class PredictionResult(BaseModel):
    label:      str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ─── /analyze-image ───────────────────────────────────────────────────────────

class AnalyzeImageResponse(BaseModel):
    food:           str
    confidence:     float
    calories:       float
    protein:        float
    carbs:          float
    fat:            float
    top_predictions: List[PredictionResult]
    serving_g:      Optional[float] = 100.0


# ─── /log-food ────────────────────────────────────────────────────────────────

class LogFoodRequest(BaseModel):
    food:       str
    confidence: float = 1.0
    calories:   float
    protein:    float
    carbs:      float
    fat:        float
    notes:      Optional[str] = None
    log_date:   Optional[date] = None


class LogFoodResponse(BaseModel):
    id:         int
    food:       str
    calories:   float
    protein:    float
    carbs:      float
    fat:        float
    log_date:   date
    logged_at:  datetime
    notes:      Optional[str]

    class Config:
        from_attributes = True


# ─── /daily-summary ───────────────────────────────────────────────────────────

class DailySummaryResponse(BaseModel):
    date:            str
    meals:           int
    total_calories:  float
    total_protein:   float
    total_carbs:     float
    total_fat:       float


# ─── /logs ────────────────────────────────────────────────────────────────────

class LogsResponse(BaseModel):
    date:  str
    count: int
    logs:  List[LogFoodResponse]
