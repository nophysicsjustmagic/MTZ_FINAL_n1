import os
from app.data.generators import DataGenerator
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from app.database.session import SessionLocal  # Правильный относительный импорт
from app.database.models import TrainingHistory  # Импорт модели
from app.ml.trainers import ModelTrainer
from app.ml.visualisers import Visualizer
from app.data.processors import DataProcessor
from app.config import DATA_DIR, MODEL_PATH, HISTORY_PATH
import multipart
import os
import json
import io
from app.data.generators import DataGenerator
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from app.database.session import SessionLocal  # Правильный относительный импорт
from app.database.models import TrainingHistory  # Импорт модели
from app.ml.trainers import ModelTrainer
from app.ml.visualisers import Visualizer
from app.data.processors import DataProcessor
from app.config import DATA_DIR, MODEL_PATH, HISTORY_PATH
import multipart
import os
import json
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Agg")
from scipy import interpolate, stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf
from keras import layers, models, callbacks, losses

import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

# SQLAlchemy для лёгкой базы данных (SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# ------------------------------------------------
# Инициализация FastAPI и создание эндпоинтов
# ------------------------------------------------

app = FastAPI(title="Сервис обработки данных и обучения модели МТЗ")

# Генерируем и сохраняем основной набор данных (один раз при запуске)
DATA_FILE = os.path.join(DATA_DIR, "synthetic_mtz_data.csv")
if not os.path.exists(DATA_FILE):

    for i in range(8):
        df_init = DataGenerator.generate_synthetic_data(random_state = 42+(10*i))
        DATA_FILE_TMP = os.path.join(DATA_DIR, f"synthetic_mtz_data{i}.csv")
        df_init.to_csv(f'{DATA_FILE_TMP}', index=False)
        print(f"Синтетические данные набора {i} сохранены в {DATA_FILE_TMP}")
        df_init.to_csv(f'{DATA_FILE}', index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Agg")
from scipy import interpolate, stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf
from keras import layers, models, callbacks, losses

import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

# SQLAlchemy для лёгкой базы данных (SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# ------------------------------------------------
# Инициализация FastAPI и создание эндпоинтов
# ------------------------------------------------

app = FastAPI(title="Сервис обработки данных и обучения модели МТЗ")

# Генерируем и сохраняем основной набор данных (один раз при запуске)
DATA_FILE = os.path.join(DATA_DIR, "synthetic_mtz_data.csv")
if not os.path.exists(DATA_FILE):

    for i in range(8):
        df_init = DataGenerator.generate_synthetic_data(random_state = 42+(10*i))
        DATA_FILE_TMP = os.path.join(DATA_DIR, f"synthetic_mtz_data{i}.csv")
        df_init.to_csv(f'{DATA_FILE_TMP}', index=False)
        print(f"Синтетические данные набора {i} сохранены в {DATA_FILE_TMP}")
        df_init.to_csv(f'{DATA_FILE}', index=False)




@app.get("/")
def read_root():
    return {"message": "Сервис обработки данных МТЗ работает!"}


@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """Загружает CSV-файл с данными и сохраняет его в папке DATA_DIR."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Требуется CSV-файл")
    file_location = os.path.join(DATA_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"message": f"Файл сохранён: {file_location}"}


@app.get("/get_data")
def get_data(filename: str):
    """Возвращает загруженный CSV-файл."""
    file_location = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(file_location, media_type='text/csv', filename=filename)


@app.post("/train_model")
def train_model(file: str = Form(None), retrain: bool = Form(False)):
    """
    Обучает модель по данным:
    - Если указан параметр file, используется указанный CSV-файл из DATA_DIR.
    - Иначе используется synthetic_mtz_data.csv.
    После обучения сохраняется история, а также в базу данных добавляется запись о тренировке.
    """
    file_path = os.path.join(DATA_DIR, file) if file else DATA_FILE
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл для обучения не найден")

    df = DataProcessor.load_and_prepare_data(file_path)
    df_filtered = DataProcessor.filter_outliers(df)
    model, history_data, eval_data = ModelTrainer.build_and_train_model(
        df_filtered, retrain=retrain
    )

    X_test, y_test, y_pred_prob = eval_data
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Сохраняем запись в базу данных
    db = SessionLocal()
    history_record = TrainingHistory(
        event="train",
        description="Модель обучена на данных: " + os.path.basename(file_path),
        accuracy=test_accuracy,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1_score=f1,
        params=json.dumps(history_data) if history_data else "{}"
    )
    db.add(history_record)
    db.commit()
    db.refresh(history_record)
    db.close()

    response = {
        "message": "Модель обучена",
        "test_accuracy": test_accuracy,
        "metrics": {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "history_saved": os.path.exists(HISTORY_PATH),
        "db_record_id": history_record.id
    }
    return JSONResponse(content=response)


@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    """
    Получает CSV-файл с данными, проводит предсказание с использованием загруженной модели
    и возвращает результат.
    """
    if file:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Требуется CSV-файл")
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode()))
    else:
        raise HTTPException(status_code=400, detail="Не переданы данные для предсказания")

    # Автоматическое определение признаков
    feature_cols = []
    for col_group in [['resistance', 'phase'], ['Re(Z)', 'Im(Z)']]:
        if all(c in df.columns for c in col_group):
            feature_cols = col_group
            break

    if not feature_cols:
        raise HTTPException(400, "Нет подходящих признаков для предсказания")

    features = df[feature_cols].values
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Модель не найдена. Обучите модель сначала.")
    model = ModelTrainer.load_saved_model(MODEL_PATH)
    y_pred_prob = model.predict(features).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    result = {
        "predictions": y_pred.tolist(),
        "probabilities": y_pred_prob.tolist()
    }
    return JSONResponse(content=result)


@app.post("/predict_visual")
async def predict_visual(file: UploadFile = File(...)):
    """
    Получает CSV-файл с данными, проводит предсказание с использованием загруженной модели,
    строит графики (распределение предсказанных вероятностей и ROC-кривую)
    и возвращает их в виде изображения PNG.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Требуется CSV-файл")
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode()))

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Модель не найдена. Обучите модель сначала.")
    model = ModelTrainer.load_saved_model(MODEL_PATH)

    features = df[['resistance', 'phase']].values
    y_pred_prob = model.predict(features).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Проверяем наличие истинных меток
    true_labels = None
    if 'impedance' in df.columns:
        bin_labels = (df['impedance'] > 0).astype(int).values
        if len(np.unique(bin_labels)) == 2:
            true_labels = bin_labels

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # График предсказанных вероятностей
    axs[0].plot(y_pred_prob, 'o-', label='Вероятность предсказания')
    axs[0].set_title("Предсказанные вероятности")
    axs[0].set_xlabel("Образец")
    axs[0].set_ylabel("Вероятность")
    axs[0].legend()

    # ROC-кривая (если есть истинные метки)
    if true_labels is not None:
        fpr, tpr, _ = roc_curve(true_labels, y_pred_prob)
        auc_val = roc_auc_score(true_labels, y_pred_prob)
        axs[1].plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.2f})')
        axs[1].plot([0, 1], [0, 1], 'k--')
        axs[1].set_title("ROC-кривая")
        axs[1].set_xlabel("False Positive Rate")
        axs[1].set_ylabel("True Positive Rate")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "Истинные метки отсутствуют или не бинарны",
                    horizontalalignment='center', verticalalignment='center')
        axs[1].set_title("ROC-кривая")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/visualize_full")
def visualize_full(file: str = None):
    """
    Выполняет полный конвейер (загрузка/подготовка, фильтрация, обучение)
    и возвращает итоговую визуализацию (6 подграфиков) в виде одного изображения PNG.
    Если указан параметр file, используется этот CSV-файл, иначе synthetic_mtz_data.csv.
    """
    history_data = None
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history_data = json.load(f)

    if history_data is None or 'loss' not in history_data:
        print("⚠ История обучения не найдена, визуализация обучения может быть неполной")

    file_path = os.path.join(DATA_DIR, file) if file else DATA_FILE
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")

    # Шаг 1: Загрузка и подготовка
    df, df_start = DataProcessor.load_and_prepare_data(file_path, NeedPlotImport=True)

    # Шаг 2: Фильтрация выбросов
    df_filtered = DataProcessor.filter_outliers(df)

    # Шаг 3: Сглаживание и интерполяция
    x_orig, y_orig, x_smooth, y_smooth = DataProcessor.smooth_and_interpolate(df_filtered, col='impedance')

    # Шаг 4: Загрузка/обучение модели и получение истории + eval_data
    model, _, eval_data = ModelTrainer.build_and_train_model(df_filtered)

    # Шаг 5: Визуализируем
    buf1, _ = Visualizer.visualize_results(
        df, df_filtered, x_orig, y_orig, x_smooth, y_smooth,
        history_data, eval_data, df_start
    )

    # Возвращаем изображение с 4 подграфиками
    return StreamingResponse(buf1, media_type="image/png")


@app.get("/get_history")
def get_history():
    """
    Возвращает историю обучения из сохранённого файла и базы данных.
    """
    db = SessionLocal()
    records = db.query(TrainingHistory).all()
    db.close()
    db_history = [
        {
            "id": rec.id,
            "event": rec.event,
            "description": rec.description,
            "accuracy": rec.accuracy,
            "roc_auc": rec.roc_auc,
            "precision": rec.precision,
            "recall": rec.recall,
            "f1_score": rec.f1_score,
            "params": json.loads(rec.params)
        }
        for rec in records
    ]
    file_history = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            file_history = json.load(f)
    return JSONResponse(content={"db_history": db_history, "file_history": file_history})
