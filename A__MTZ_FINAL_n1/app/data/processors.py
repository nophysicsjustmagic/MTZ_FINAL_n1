from scipy.constants import mu_0
import matplotlib
import matplotlib.pyplot as plt
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
import io


import numpy as np
from numpy import complexfloating
import pandas as pd

class DataProcessor:

    @staticmethod
    def _visualize_getten_data(df: pd.DataFrame):
        plt.figure(figsize=(18, 14))
        TmpIndex = 1
        for NameQuantity in ['impedance', 'resistance', 'phase']:
            plt.subplot(3, 2, TmpIndex)
            plt.plot(df['x'], df[f'{NameQuantity}'], 'o', markersize=3, label='Исходные данные')
            plt.plot(df['x'], df[f'{NameQuantity}'], '-', linewidth=2, label='Сглаженная кривая')
            plt.xlabel('x')
            plt.ylabel(f'{NameQuantity}')
            plt.title('Исходные данные')
            plt.legend()
            TmpIndex += 1
        buf0 = io.BytesIO()
        plt.savefig(buf0, format='png')
        buf0.seek(0)
        plt.close('all')
        return buf0



    @staticmethod
    def _calculate_phase_resistance(df: pd.DataFrame):
        """Рассчитывает phase и resistance из Re(Z) и Im(Z) при необходимости"""
        if 'Re(Z)' in df.columns and 'Im(Z)' in df.columns:
            df['resistance'] = ( (df['Re(Z)'] ** 2) + (df['Im(Z)'] ** 2) ) * (df['x']/(2*np.pi*mu_0))
            df['phase'] = np.degrees(np.arctan2(df['Im(Z)'], df['Re(Z)']))
            df['impedance'] = np.abs( df['Re(Z)'] + (1j*df['Im(Z)']) )
            df.drop(['Re(Z)', 'Im(Z)'], axis=1, inplace=True)
        else:
            df['resistance'] = (df['impedance'] ** 2) * (df['x'] / (2 * np.pi * mu_0))
            #phese уже должна присутствовать
        return df

    @staticmethod
    def _validate_physical_limits(df: pd.DataFrame):
        """Фильтрация по физическим ограничениям"""
        # Фильтрация некорректных значений
                                    #############

        min_phase = 0       #   Допустимый диапазон фаз
        max_phase = 90
                                    #############
        mask = (
                (df['resistance'] > 0) &
                (df['phase'].between(min_phase, max_phase))  # Фильтрация по критерию: допустимый диапазон фаз
        )
        invalid_count = len(df) - mask.sum()
        if invalid_count > 0:
            print(f"Удалено {invalid_count} записей с физически некорректными значениями")
        return df[mask]

    @staticmethod
    def load_and_prepare_data(file_path: str, NeedPlotImport = False) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        print("Первые строки загруженных данных:")
        print(df.head())
        df_import = df.copy(deep = True)
        # Автоматическое определение типа данных
        df = DataProcessor._calculate_phase_resistance(df)
        df_import = DataProcessor._calculate_phase_resistance(df_import)
        '''
        print("Статистика данных после получения:")
        # Добавить сюда еще возможность вернуться от нормированных величин к
        pd.set_option('display.max_columns', None)  # Без ограничения колонок
        print(df_import.describe())
        '''

        if df.isnull().sum().sum() > 0:
            df.fillna(method='ffill', inplace=True)

        # Фильтрация физически некорректных значений
        df = DataProcessor._validate_physical_limits(df)

        # Нормализация
        for col in ['impedance', 'resistance', 'phase']:
            if col in df.columns:
                df[f'{col}_Absolute'] = df[col]
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        print("Статистика данных после нормализации:")
        #Добавить сюда еще возможность вернуться от нормированных величин к
        pd.set_option('display.max_columns', None)  # Без ограничения колонок
        print(df.describe())
        print("\nСама таблица:\n")
        print(df.head())

        if NeedPlotImport:
            return df, df_import
        else:
            return df

    @staticmethod
    def filter_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
        """Автоматический выбор колонок для анализа"""
        if cols is None:
            cols = [c for c in ['impedance', 'resistance', 'phase'] if c in df.columns]

        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[cols])
        df_filtered = df[df['anomaly'] == 1].copy()
        print(f"Удалено {df.shape[0] - df_filtered.shape[0]} выбросов методом Isolation Forest")

        z_scores = np.abs(stats.zscore(df_filtered[cols]))
        mask = (z_scores < 2.5).all(axis=1)
        df_filtered = df_filtered[mask].copy()
        print(f"Осталось {df_filtered.shape[0]} точек после фильтрации по z-score")

        return df_filtered

    @staticmethod
    def smooth_and_interpolate(df: pd.DataFrame, col='impedance', smooth_param=5):
        df_sorted = df.sort_values('x')
        x = df_sorted['x'].values
        y = df_sorted[col].values

        spline = interpolate.UnivariateSpline(x, y, s=smooth_param)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = spline(x_smooth)

        return x, y, x_smooth, y_smooth