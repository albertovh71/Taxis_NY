"""Entrenar Linear Regression y XGBoost con Yellow + FHVHV (2023-2025)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def train_and_evaluate(
    input_path: str = 'data/processed/yellow_taxi_features_combined.parquet',
    test_date: str = '2025-01-01',
) -> dict:
    """Entrenar Linear Regression y XGBoost, evaluar en test set temporal.

    Split: train < test_date, test >= test_date
    """
    print('[train] cargando datos')
    df = pd.read_parquet(input_path)

    # Features
    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'hour_of_week',
        'is_holiday', 'is_holiday_eve',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_3h', 'rolling_mean_12h', 'rolling_mean_24h', 'rolling_mean_168h'
    ]

    target = 'trip_count'

    # Split temporal
    test_date_dt = pd.to_datetime(test_date)
    train = df[df['pickup_datetime'] < test_date_dt].copy()
    test = df[df['pickup_datetime'] >= test_date_dt].copy()

    print(f'[split] train: {len(train):,} filas, test: {len(test):,} filas')
    print(f'[split] train dates: {train["pickup_datetime"].min()} a {train["pickup_datetime"].max()}')
    print(f'[split] test dates: {test["pickup_datetime"].min()} a {test["pickup_datetime"].max()}')

    # Preparar X, y
    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_test = test[feature_cols].fillna(0)
    y_test = test[target]

    print(f'\n[train] X shape: {X_train.shape}, y shape: {y_train.shape}')
    print(f'[test] X shape: {X_test.shape}, y shape: {y_test.shape}')

    results = {}

    # ==================== Linear Regression ====================
    print('\n' + '='*60)
    print('LINEAR REGRESSION')
    print('='*60)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f'MAE:  {mae_lr:.4f}')
    print(f'RMSE: {rmse_lr:.4f}')
    print(f'R2:   {r2_lr:.4f}')

    results['linear'] = {
        'model': lr,
        'y_pred': y_pred_lr,
        'mae': mae_lr,
        'rmse': rmse_lr,
        'r2': r2_lr,
    }

    # ==================== XGBoost ====================
    print('\n' + '='*60)
    print('XGBOOST')
    print('='*60)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        device='cpu',
        verbose=False,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)

    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f'MAE:  {mae_xgb:.4f}')
    print(f'RMSE: {rmse_xgb:.4f}')
    print(f'R2:   {r2_xgb:.4f}')

    results['xgboost'] = {
        'model': xgb_model,
        'y_pred': y_pred_xgb,
        'mae': mae_xgb,
        'rmse': rmse_xgb,
        'r2': r2_xgb,
    }

    # ==================== Comparacion ====================
    print('\n' + '='*60)
    print('COMPARACION')
    print('='*60)
    print(f'{"Metrica":<10} {"Linear":<12} {"XGBoost":<12} {"Mejora (%)"}')
    print('-'*46)

    mae_improvement = 100 * (mae_lr - mae_xgb) / mae_lr
    rmse_improvement = 100 * (rmse_lr - rmse_xgb) / rmse_lr
    r2_improvement = 100 * (r2_xgb - r2_lr) / abs(r2_lr) if r2_lr != 0 else 0

    print(f'{"MAE":<10} {mae_lr:<12.4f} {mae_xgb:<12.4f} {mae_improvement:+.1f}%')
    print(f'{"RMSE":<10} {rmse_lr:<12.4f} {rmse_xgb:<12.4f} {rmse_improvement:+.1f}%')
    print(f'{"R2":<10} {r2_lr:<12.4f} {r2_xgb:<12.4f} {r2_improvement:+.1f}%')

    if mae_xgb < mae_lr:
        print(f'\n[WINNER] XGBoost es mejor: MAE {mae_xgb:.4f} vs {mae_lr:.4f}')
    else:
        print(f'\n[WINNER] Linear es mejor: MAE {mae_lr:.4f} vs {mae_xgb:.4f}')

    # Feature importance (XGBoost)
    print('\n' + '='*60)
    print('TOP 10 FEATURES (XGBoost)')
    print('='*60)
    importance = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f'{row["feature"]:<20} {row["importance"]:.4f}')

    return results


if __name__ == '__main__':
    results = train_and_evaluate()
