"""Afinar Linear Regression: regularización y escalado."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def tune_linear(
    input_path: str = 'data/processed/yellow_taxi_features.parquet',
    test_date: str = '2025-03-01',
) -> dict:
    """Probar variantes de Linear: regularización, escalado, feature selection."""

    print('[load] cargando datos')
    df = pd.read_parquet(input_path)

    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'hour_of_week',
        'is_holiday', 'is_holiday_eve',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_3h', 'rolling_mean_12h', 'rolling_mean_24h', 'rolling_mean_168h'
    ]
    target = 'trip_count'

    test_date_dt = pd.to_datetime(test_date)
    train = df[df['pickup_datetime'] < test_date_dt].copy()
    test = df[df['pickup_datetime'] >= test_date_dt].copy()

    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_test = test[feature_cols].fillna(0)
    y_test = test[target]

    print(f'[split] train: {len(train):,}, test: {len(test):,}')

    results = {}

    # ==================== 1. Linear base (sin escalado) ====================
    print('\n' + '='*60)
    print('1. LINEAR REGRESSION (baseline)')
    print('='*60)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'MAE:  {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2:   {r2:.6f}')
    results['Linear (base)'] = {'mae': mae, 'rmse': rmse, 'r2': r2}

    # ==================== 2. Linear con StandardScaler ====================
    print('\n' + '='*60)
    print('2. LINEAR + StandardScaler')
    print('='*60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_scaled = LinearRegression()
    lr_scaled.fit(X_train_scaled, y_train)
    y_pred = lr_scaled.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'MAE:  {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2:   {r2:.6f}')
    results['Linear + Scaler'] = {'mae': mae, 'rmse': rmse, 'r2': r2}

    # ==================== 3. Ridge con varios alphas ====================
    print('\n' + '='*60)
    print('3. RIDGE (L2 regularization)')
    print('='*60)
    best_ridge_mae = float('inf')
    best_ridge_alpha = None

    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f'alpha={alpha:<6}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.6f}')

        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_ridge_alpha = alpha
            best_ridge_metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

    print(f'[best] alpha={best_ridge_alpha}: MAE={best_ridge_mae:.4f}')
    results[f'Ridge (alpha={best_ridge_alpha})'] = best_ridge_metrics

    # ==================== 4. Lasso con varios alphas ====================
    print('\n' + '='*60)
    print('4. LASSO (L1 regularization)')
    print('='*60)
    best_lasso_mae = float('inf')
    best_lasso_alpha = None

    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        try:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_train_scaled, y_train)
            y_pred = lasso.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f'alpha={alpha:<6}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.6f}')

            if mae < best_lasso_mae:
                best_lasso_mae = mae
                best_lasso_alpha = alpha
                best_lasso_metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
        except:
            print(f'alpha={alpha:<6}: convergencia fallida')

    if best_lasso_alpha:
        print(f'[best] alpha={best_lasso_alpha}: MAE={best_lasso_mae:.4f}')
        results[f'Lasso (alpha={best_lasso_alpha})'] = best_lasso_metrics

    # ==================== 5. ElasticNet ====================
    print('\n' + '='*60)
    print('5. ElasticNet (L1+L2)')
    print('='*60)
    best_elastic_mae = float('inf')
    best_elastic_alpha = None

    for alpha in [0.001, 0.01, 0.1]:
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        elastic.fit(X_train_scaled, y_train)
        y_pred = elastic.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f'alpha={alpha:<6}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.6f}')

        if mae < best_elastic_mae:
            best_elastic_mae = mae
            best_elastic_alpha = alpha
            best_elastic_metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

    print(f'[best] alpha={best_elastic_alpha}: MAE={best_elastic_mae:.4f}')
    results[f'ElasticNet (alpha={best_elastic_alpha})'] = best_elastic_metrics

    # ==================== RESUMEN ====================
    print('\n' + '='*60)
    print('RANKING FINAL')
    print('='*60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])

    print(f'{"Modelo":<30} {"MAE":<10} {"RMSE":<10} {"R2":<10}')
    print('-'*60)
    for name, metrics in sorted_results:
        print(f'{name:<30} {metrics["mae"]:<10.4f} {metrics["rmse"]:<10.4f} {metrics["r2"]:<10.6f}')

    winner_name, winner_metrics = sorted_results[0]
    print(f'\n[WINNER] {winner_name}: MAE={winner_metrics["mae"]:.4f}')

    return results


if __name__ == '__main__':
    tune_linear()
