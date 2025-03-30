import os
import numpy as np
import pandas as pd
from scipy.constants import mu_0


class DataGenerator:
    """Класс для генерации синтетических данных (основных и альтернативных)."""

    @staticmethod
    def generate_synthetic_data(num_points=500, noise_level=0.05,
                                outlier_fraction=0.05, random_state=42, add_impedance_components = False):
        """
        Генерирует синтетические данные для одной точки измерения.
        Каждый набор содержит 8 наборов кривых (например, измерение импеданса),
        к которым добавлен шум и выбросы.
        """
        np.random.seed(random_state)
        x = np.linspace(0, 10, num_points)
        true_curve = np.exp(-x / 5) * 100

        data_sets = []
        for i in range(8):
            noise = np.random.normal(0, noise_level * np.mean(true_curve), size=num_points)
            curve = true_curve + noise
            data_sets.append(curve)

        records = []
        for i in range(num_points):
            impedance_values = [ds[i] for ds in data_sets]
            impedance_mean = np.mean(impedance_values)
            resistance = (impedance_mean**2) * (x[i] / (2*np.pi*mu_0))

            phase = 45 + np.random.normal(0, 2)
            records.append({
                'x': x[i],
                'impedance': impedance_mean,
                'resistance': resistance,
                'phase': phase
            })

        df = pd.DataFrame(records)
        # Добавляем выбросы
        num_outliers = int(outlier_fraction * num_points)
        outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
        for idx in outlier_indices:
            df.loc[idx, 'impedance'] *= np.random.uniform(1.2, 1.5)
            df.loc[idx, 'resistance'] *= np.random.uniform(0.5, 0.8)
            df.loc[idx, 'phase'] += np.random.uniform(5, 25)
        if add_impedance_components:
            # Генерация Re(Z) и Im(Z) вместо resistance и phase
            df['Re(Z)'] = df['impedance'] * np.cos(np.radians(df['phase']))
            df['Im(Z)'] = df['impedance'] * np.sin(np.radians(df['phase']))
            df.drop(['impedance', 'phase'], axis=1, inplace=True)
        return df

    @staticmethod
    def generate_alternative_data(num_points=500, noise_level=0.1,
                                  outlier_fraction=0.1, random_state=24):
        """
        Генерирует альтернативный синтетический набор данных
        с иными параметрами шума и выбросов.
        """
        return DataGenerator.generate_synthetic_data(
            num_points=num_points,
            noise_level=noise_level,
            outlier_fraction=outlier_fraction,
            random_state=random_state,
            add_impedance_components=True
        )
