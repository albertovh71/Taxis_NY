"""Entrenar y serializar modelo Linear Regression (Yellow + FHVHV) para produccion."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


def save_model():
    """Entrenar LinearRegression con Yellow + FHVHV y guardar con joblib."""

    print('[model] cargando datos')
    df = pd.read_parquet('data/processed/yellow_taxi_features_combined.parquet')

    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'hour_of_week',
        'is_holiday', 'is_holiday_eve',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_3h', 'rolling_mean_12h', 'rolling_mean_24h', 'rolling_mean_168h'
    ]
    target = 'trip_count'

    X = df[feature_cols].fillna(0)
    y = df[target]

    print(f'[train] X: {X.shape}, y: {y.shape}')

    # Entrenar con todos los datos
    model = LinearRegression()
    model.fit(X, y)

    # Guardar modelo
    Path('models').mkdir(exist_ok=True)
    model_path = Path('models/linear_model_combined.joblib')
    joblib.dump(model, model_path)
    print(f'[save] modelo guardado en {model_path}')

    # Guardar ultimos 14 dias para lags
    print('[history] guardando ultimos 14 dias para calcular lags')
    last_date = df['pickup_datetime'].max()
    cutoff_date = last_date - pd.Timedelta(days=14)

    history_df = df[df['pickup_datetime'] >= cutoff_date].copy()
    history_df = history_df[['PULocationID', 'pickup_datetime', 'trip_count']].sort_values(
        ['PULocationID', 'pickup_datetime']
    )

    history_path = Path('data/processed/history_recent_combined.parquet')
    history_df.to_parquet(history_path, index=False)
    print(f'[history] {len(history_df):,} registros guardados en {history_path}')

    print('\n[done] modelo y historico listos para produccion')


if __name__ == '__main__':
    save_model()
