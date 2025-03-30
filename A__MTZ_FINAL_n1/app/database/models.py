# SQLAlchemy для лёгкой базы данных (SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

class TrainingHistory(Base):
    __tablename__ = "training_history"
    id = Column(Integer, primary_key=True, index=True)
    event = Column(String, index=True)  # например, "train", "update"
    description = Column(Text, nullable=True)  # комментарий или описание
    accuracy = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    params = Column(Text, nullable=True)  # параметры тренировки, JSON


