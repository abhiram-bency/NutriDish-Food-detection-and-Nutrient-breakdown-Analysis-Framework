"""
NutriDish — Database Layer (SQLite via SQLAlchemy)
"""

import os
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Date, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ─── Setup ────────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("NUTRIDISH_DB", "nutridish.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Models ───────────────────────────────────────────────────────────────────

class FoodLog(Base):
    __tablename__ = "food_logs"

    id          = Column(Integer, primary_key=True, index=True)
    food        = Column(String, nullable=False, index=True)
    confidence  = Column(Float, nullable=False)
    calories    = Column(Float, nullable=False)
    protein     = Column(Float, nullable=False)
    carbs       = Column(Float, nullable=False)
    fat         = Column(Float, nullable=False)
    log_date    = Column(Date, nullable=False, default=date.today, index=True)
    logged_at   = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes       = Column(String, nullable=True)


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── CRUD helpers ─────────────────────────────────────────────────────────────

def create_food_log(
    db: Session,
    food: str,
    confidence: float,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    notes: Optional[str] = None,
    log_date: Optional[date] = None,
) -> FoodLog:
    entry = FoodLog(
        food=food,
        confidence=confidence,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        notes=notes,
        log_date=log_date or date.today(),
        logged_at=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_daily_logs(db: Session, target_date: Optional[date] = None) -> List[FoodLog]:
    target = target_date or date.today()
    return db.query(FoodLog).filter(FoodLog.log_date == target).all()


def get_daily_summary(db: Session, target_date: Optional[date] = None) -> dict:
    target = target_date or date.today()
    rows = (
        db.query(
            func.sum(FoodLog.calories).label("total_calories"),
            func.sum(FoodLog.protein).label("total_protein"),
            func.sum(FoodLog.carbs).label("total_carbs"),
            func.sum(FoodLog.fat).label("total_fat"),
            func.count(FoodLog.id).label("meals"),
        )
        .filter(FoodLog.log_date == target)
        .one()
    )
    return {
        "date": str(target),
        "meals": rows.meals or 0,
        "total_calories": round(rows.total_calories or 0, 1),
        "total_protein":  round(rows.total_protein  or 0, 1),
        "total_carbs":    round(rows.total_carbs    or 0, 1),
        "total_fat":      round(rows.total_fat      or 0, 1),
    }
