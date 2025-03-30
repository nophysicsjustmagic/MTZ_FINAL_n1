from scipy.interpolate import UnivariateSpline
import matplotlib
import matplotlib.pyplot as plt
import io
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
import numpy as np


class Visualizer:
    """Класс для построения и сохранения визуализаций."""

    @staticmethod
    def visualize_results(df_prepared, df_filtered, x_orig, y_orig, x_smooth, y_smooth,
                          history, eval_data, df_start):
        """
        Строит графики:
        - Исходная и сглаженная кривая импеданса
        - Распределение параметров до и после фильтрации
        - Графики обучения (если история доступна)
        - ROC-кривая
        Возвращает два буфера (с 7 подграфиками и с ROC-кривой).
        """
        X_test, y_test, y_pred_prob = eval_data

        # Первая фигура исходные данные(3 подграфика), данные после обработки(4 подграфика)
        plt.figure(figsize=(18, 14))

        TmpIndex = 1
        for NameQuantity in ['impedance', 'resistance', 'phase']:
            if TmpIndex == 4:
                break
            plt.subplot(3, 3, TmpIndex)

            # Исходные данные
            x = df_start['x'].values
            y = df_start[f'{NameQuantity}'].values

            # Сглаживающий сплайн
            spline = UnivariateSpline(x, y, s=len(x) *10)  # Увеличенный параметр сглаживания
            x_smooth_tmp = np.linspace(x.min(), x.max(), 500)
            y_smooth_tmp = spline(x_smooth_tmp)

            # Построение
            plt.plot(x, y, 'o', color = 'blue', markersize=3, label='Исходные данные', alpha=0.5)
            plt.plot(x_smooth_tmp, y_smooth_tmp, '-', color = 'orange', linewidth=2, label='Сглаженный тренд')

            plt.xlabel('x')
            plt.ylabel(f'{NameQuantity}')
            plt.title(f'Сглаживание {NameQuantity}')
            plt.legend()
            TmpIndex += 1
        
        
        # (1) Исходная vs сглаженная кривая
        plt.subplot(3, 3, 4)
        plt.plot(x_orig, y_orig, 'o', markersize=3, label='Исходные данные')
        plt.plot(x_smooth, y_smooth, '-', linewidth=2, label='Сглаженная кривая')
        plt.xlabel('x')
        plt.ylabel('Нормированный импеданс')
        plt.title('Сплайн-интерполяция импеданса')
        plt.legend()

        # (1, 2, 3) Распределение до/после фильтрации для сопротивления
        TmpIndex = 5
        for NameQuantity in ['impedance', 'resistance', 'phase']:
            plt.subplot(3, 3, TmpIndex)
            plt.scatter(df_prepared['x'], df_prepared[f'{NameQuantity}_Absolute'],
                        c='gray', alpha=0.5, label='Выбросы')
            plt.scatter(df_filtered['x'], df_filtered[f'{NameQuantity}_Absolute'],
                        c='blue', alpha=0.8, label='После фильтрации')
            plt.xlabel('x')
            plt.ylabel(f'{NameQuantity}')
            plt.title('Фильтрация выбросов')
            plt.legend()
            TmpIndex += 1




        # (4) График потерь
        plt.subplot(3, 3, 8)
        if history is not None and 'loss' in history:
            plt.plot(history['loss'], label='Обучение (loss)')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Валидация (loss)')
            plt.xlabel('Эпоха')
            plt.ylabel('Потеря')
            plt.title('График потерь')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'История обучения не доступна',
                     horizontalalignment='center', verticalalignment='center')

        # (5) График точности
        plt.subplot(3, 3, 9)
        if history is not None and 'accuracy' in history:
            plt.plot(history['accuracy'], label='Обучение (accuracy)')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Валидация (accuracy)')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность')
            plt.title('График точности')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'История обучения не доступна',
                     horizontalalignment='center', verticalalignment='center')

        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)

        # Вторая фигура: ROC-кривая
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_val = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {auc_val:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plt.close('all')

        return buf1, buf2