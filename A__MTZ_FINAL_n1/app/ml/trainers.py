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
import json
from keras import layers, models, callbacks, losses
import os
from app.config import MODEL_PATH,HISTORY_PATH
import pandas as pd



class ModelTrainer:
    """Класс для построения, обучения и оценки нейронной сети."""

    @staticmethod
    def custom_loss(y_true, y_pred):
        koef = 0.3
        bce_loss = losses.binary_crossentropy(y_true, y_pred)

        # Исправляем расчет разностей
        y_pred = tf.reshape(y_pred, [-1])  # Делаем y_pred одномерным

        diff = tf.concat([[0], y_pred[1:] - y_pred[:-1]], axis=0)  # Добавляем 0 для выравнивания формы
        penalty = tf.reduce_mean(tf.nn.relu(diff))  # Штраф за возрастание

        return bce_loss + koef * penalty

    @staticmethod
    def load_saved_model(model_path):
        return models.load_model(
            model_path,
            custom_objects={"custom_loss": ModelTrainer.custom_loss}
        )

    @staticmethod
    def build_deep_model(input_shape: int):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss=ModelTrainer.custom_loss, metrics=['accuracy'])

        return model

    @staticmethod
    def build_and_train_model(df: pd.DataFrame, model_path=MODEL_PATH, retrain=False):
        features = df[['resistance', 'phase']].values
        labels = (df['impedance'] > 0).astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        if os.path.exists(model_path) and not retrain:
            print("Загрузка сохраненной модели...")
            model = ModelTrainer.load_saved_model(model_path)
            history_data = None
        else:
            model = ModelTrainer.build_deep_model(X_train.shape[1])
            checkpoint = callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', save_best_only=True, verbose=1
            )
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
            )

            print("Начало обучения модели...")
            history = model.fit(
                X_train, y_train,
                epochs=100, batch_size=16,
                validation_split=0.1,
                callbacks=[checkpoint, early_stop],
                verbose=1
            )

            history_data = history.history
            with open(HISTORY_PATH, "w") as f:
                json.dump(history_data, f)
            print("Обучение завершено и история сохранена.")

            # Сохранение модели с поддержкой custom_loss
            model.save(model_path, save_format="h5")

        return model, history_data, (X_test, y_test, model.predict(X_test).ravel())