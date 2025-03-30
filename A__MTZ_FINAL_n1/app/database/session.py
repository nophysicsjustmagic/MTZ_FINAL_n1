import os
# SQLAlchemy для лёгкой базы данных (SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


# Создадим папки для хранения данных и модели
DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")

# Настройка базы данных SQLite
DATABASE_URL = "sqlite:///./history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)