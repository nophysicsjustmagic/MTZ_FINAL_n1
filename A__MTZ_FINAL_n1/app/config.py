import os

# Директории
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "model")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Пути к файлам
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")


# Настройки БД
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./history.db")