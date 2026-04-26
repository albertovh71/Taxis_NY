"""Evaluar modelo en train, test y futuro con análisis por zona y hora."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcular MAPE, ignorando casos donde y_true=0."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def evaluate_period(
    y_true: pd.Series,
    y_pred: np.ndarray,
    period_name: str,
) -> dict:
    """Calcular métricas para un período."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true.values, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'period': period_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'n_samples': len(y_true),
    }


def evaluate_by_zone(
    df_period: pd.DataFrame,
    y_pred: np.ndarray,
    period_name: str,
) -> pd.DataFrame:
    """Calcular métricas por zona."""
    df = df_period.copy()
    df['y_pred'] = y_pred

    zone_metrics = []

    for zone_id in df['PULocationID'].unique():
        zone_data = df[df['PULocationID'] == zone_id]

        y_true = zone_data['trip_count'].values
        y_pred_zone = zone_data['y_pred'].values

        if len(y_true) == 0:
            continue

        mae = mean_absolute_error(y_true, y_pred_zone)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_zone))
        mape = calculate_mape(y_true, y_pred_zone)

        zone_metrics.append({
            'zone': zone_id,
            'period': period_name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_samples': len(y_true),
            'mean_demand': y_true.mean(),
        })

    return pd.DataFrame(zone_metrics)


def evaluate_by_hour(
    df_period: pd.DataFrame,
    y_pred: np.ndarray,
    period_name: str,
) -> pd.DataFrame:
    """Calcular métricas por hora del día."""
    df = df_period.copy()
    df['y_pred'] = y_pred

    hour_metrics = []

    for hour in sorted(df['hour'].unique()):
        hour_data = df[df['hour'] == hour]

        y_true = hour_data['trip_count'].values
        y_pred_hour = hour_data['y_pred'].values

        if len(y_true) == 0:
            continue

        mae = mean_absolute_error(y_true, y_pred_hour)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_hour))
        mape = calculate_mape(y_true, y_pred_hour)

        hour_metrics.append({
            'hour': hour,
            'period': period_name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_samples': len(y_true),
            'mean_demand': y_true.mean(),
        })

    return pd.DataFrame(hour_metrics)


def main():
    """Evaluar modelo en train, test y futuro."""
    print('='*70)
    print('EVALUACIÓN COMPREHENSIVA DEL MODELO')
    print('='*70)

    # Cargar datos
    print('\n[load] cargando datos procesados...')
    df = pd.read_parquet('data/processed/yellow_taxi_features.parquet')
    print(f'  total: {len(df):,} registros')
    print(f'  rango: {df["pickup_datetime"].min()} a {df["pickup_datetime"].max()}')

    # Cargar modelo
    print('\n[load] cargando modelo entrenado...')
    model = joblib.load('models/linear_model.joblib')
    print(f'  modelo: Linear Regression')

    # Features
    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'hour_of_week',
        'is_holiday', 'is_holiday_eve',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_3h', 'rolling_mean_12h', 'rolling_mean_24h', 'rolling_mean_168h'
    ]
    target = 'trip_count'

    # Definir períodos de evaluación
    test_date = pd.to_datetime('2025-03-01')
    future_date = df['pickup_datetime'].max() + timedelta(days=1)

    # TRAIN PERIOD
    print(f'\n[train] evaluando período de entrenamiento (< {test_date.date()})')
    df_train = df[df['pickup_datetime'] < test_date].copy()
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target]
    y_pred_train = model.predict(X_train)

    metrics_train = evaluate_period(y_train, y_pred_train, 'train')
    print(f'  {len(df_train):,} muestras')
    print(f'  MAE: {metrics_train["mae"]:.4f}, RMSE: {metrics_train["rmse"]:.4f}, MAPE: {metrics_train["mape"]:.2f}%')

    # TEST PERIOD
    print(f'\n[test] evaluando período de test (>= {test_date.date()})')
    df_test = df[df['pickup_datetime'] >= test_date].copy()
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test[target]
    y_pred_test = model.predict(X_test)

    metrics_test = evaluate_period(y_test, y_pred_test, 'test')
    print(f'  {len(df_test):,} muestras')
    print(f'  MAE: {metrics_test["mae"]:.4f}, RMSE: {metrics_test["rmse"]:.4f}, MAPE: {metrics_test["mape"]:.2f}%')

    # Comparación train vs test
    print('\n' + '='*70)
    print('COMPARACIÓN TRAIN vs TEST')
    print('='*70)
    print(f'{"Métrica":<10} {"Train":<15} {"Test":<15} {"Degradación (%)"}')
    print('-'*55)

    mae_degrad = 100 * (metrics_test['mae'] - metrics_train['mae']) / metrics_train['mae']
    rmse_degrad = 100 * (metrics_test['rmse'] - metrics_train['rmse']) / metrics_train['rmse']
    mape_degrad = metrics_test['mape'] - metrics_train['mape']

    print(f'{"MAE":<10} {metrics_train["mae"]:<15.4f} {metrics_test["mae"]:<15.4f} {mae_degrad:+.1f}%')
    print(f'{"RMSE":<10} {metrics_train["rmse"]:<15.4f} {metrics_test["rmse"]:<15.4f} {rmse_degrad:+.1f}%')
    print(f'{"MAPE":<10} {metrics_train["mape"]:<15.2f}% {metrics_test["mape"]:<14.2f}% {mape_degrad:+.1f} pp')
    print(f'{"R²":<10} {metrics_train["r2"]:<15.4f} {metrics_test["r2"]:<15.4f}')

    # ANÁLISIS POR ZONA
    print('\n' + '='*70)
    print('ANÁLISIS POR ZONA')
    print('='*70)

    zone_metrics_train = evaluate_by_zone(df_train, y_pred_train, 'train')
    zone_metrics_test = evaluate_by_zone(df_test, y_pred_test, 'test')
    zone_metrics_all = pd.concat([zone_metrics_train, zone_metrics_test], ignore_index=True)

    # Top 10 zonas por MAPE en test
    print('\nTop 10 zonas con mayor MAPE en TEST:')
    top_zones = zone_metrics_test.nlargest(10, 'mape')[['zone', 'mape', 'mean_demand', 'n_samples']]
    for idx, row in top_zones.iterrows():
        print(f'  zona {int(row["zone"]):3d}: MAPE={row["mape"]:6.2f}% | demanda_media={row["mean_demand"]:7.2f} | muestras={int(row["n_samples"]):4d}')

    print('\nTop 10 zonas con menor MAPE en TEST:')
    bottom_zones = zone_metrics_test.nsmallest(10, 'mape')[['zone', 'mape', 'mean_demand', 'n_samples']]
    for idx, row in bottom_zones.iterrows():
        print(f'  zona {int(row["zone"]):3d}: MAPE={row["mape"]:6.2f}% | demanda_media={row["mean_demand"]:7.2f} | muestras={int(row["n_samples"]):4d}')

    # ANÁLISIS POR HORA
    print('\n' + '='*70)
    print('ANÁLISIS POR HORA DEL DÍA')
    print('='*70)

    hour_metrics_train = evaluate_by_hour(df_train, y_pred_train, 'train')
    hour_metrics_test = evaluate_by_hour(df_test, y_pred_test, 'test')
    hour_metrics_all = pd.concat([hour_metrics_train, hour_metrics_test], ignore_index=True)

    print('\nMAPE por hora (train vs test):')
    print(f'{"Hora":<6} {"Train MAPE":<15} {"Test MAPE":<15} {"Cambio"}')
    print('-'*52)

    for hour in sorted(hour_metrics_train['hour'].unique()):
        train_row = hour_metrics_train[hour_metrics_train['hour'] == hour]
        test_row = hour_metrics_test[hour_metrics_test['hour'] == hour]

        if len(train_row) > 0 and len(test_row) > 0:
            train_mape = train_row.iloc[0]['mape']
            test_mape = test_row.iloc[0]['mape']
            change = test_mape - train_mape
            print(f'{int(hour):2d}:00  {train_mape:6.2f}%{"":<8} {test_mape:6.2f}%{"":<8} {change:+.1f}%')

    # Guardar resultados
    print('\n' + '='*70)
    print('GUARDANDO RESULTADOS')
    print('='*70)

    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)

    # Métricas globales
    metrics_df = pd.DataFrame([metrics_train, metrics_test])
    metrics_path = output_dir / 'model_metrics_global.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f'  métricas globales: {metrics_path}')

    # Métricas por zona
    zone_metrics_path = output_dir / 'model_metrics_by_zone.csv'
    zone_metrics_all.to_csv(zone_metrics_path, index=False)
    print(f'  métricas por zona: {zone_metrics_path}')

    # Métricas por hora
    hour_metrics_path = output_dir / 'model_metrics_by_hour.csv'
    hour_metrics_all.to_csv(hour_metrics_path, index=False)
    print(f'  métricas por hora: {hour_metrics_path}')

    # Predicciones detalladas
    df_with_pred = df.copy()
    df_with_pred['y_pred'] = model.predict(df_with_pred[feature_cols].fillna(0))
    df_with_pred['error'] = df_with_pred['trip_count'] - df_with_pred['y_pred']
    df_with_pred['abs_error'] = np.abs(df_with_pred['error'])

    pred_path = output_dir / 'model_predictions.parquet'
    df_with_pred.to_parquet(pred_path, index=False)
    print(f'  predicciones detalladas: {pred_path}')

    # VISUALIZACIONES
    print('\n' + '='*70)
    print('GENERANDO VISUALIZACIONES')
    print('='*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluación del Modelo: Train vs Test', fontsize=14, fontweight='bold')

    # Scatter: Actual vs Predicted (Train)
    ax = axes[0, 0]
    ax.scatter(y_train, y_pred_train, alpha=0.3, s=10)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicción')
    ax.set_title(f'Train (n={len(y_train):,})\nMAE={metrics_train["mae"]:.2f}, MAPE={metrics_train["mape"]:.1f}%')
    ax.grid(alpha=0.3)

    # Scatter: Actual vs Predicted (Test)
    ax = axes[0, 1]
    ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color='orange')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicción')
    ax.set_title(f'Test (n={len(y_test):,})\nMAE={metrics_test["mae"]:.2f}, MAPE={metrics_test["mape"]:.1f}%')
    ax.grid(alpha=0.3)

    # Residuos (Train)
    ax = axes[1, 0]
    residuals_train = y_train.values - y_pred_train
    ax.scatter(y_pred_train, residuals_train, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Residuo')
    ax.set_title('Residuos - Train')
    ax.grid(alpha=0.3)

    # Residuos (Test)
    ax = axes[1, 1]
    residuals_test = y_test.values - y_pred_test
    ax.scatter(y_pred_test, residuals_test, alpha=0.3, s=10, color='orange')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Residuo')
    ax.set_title('Residuos - Test')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'model_predictions_scatter.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f'  gráfico scatter: {plot_path}')
    plt.close()

    # MAPE por hora
    fig, ax = plt.subplots(figsize=(12, 5))

    pivot_mape = hour_metrics_all.pivot(index='hour', columns='period', values='mape')
    pivot_mape.plot(ax=ax, marker='o', linewidth=2)
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE por Hora: Train vs Test')
    ax.grid(alpha=0.3)
    ax.legend(title='Período')
    plt.xticks(range(0, 24))

    plot_path = output_dir / 'model_mape_by_hour.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f'  gráfico MAPE por hora: {plot_path}')
    plt.close()

    # Distribución de errores
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(residuals_train, bins=50, alpha=0.7, label='Train', edgecolor='black')
    ax.set_xlabel('Residuo')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Residuos - Train')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.hist(residuals_test, bins=50, alpha=0.7, label='Test', color='orange', edgecolor='black')
    ax.set_xlabel('Residuo')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Residuos - Test')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / 'model_residuals_distribution.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f'  gráfico distribución residuos: {plot_path}')
    plt.close()

    print('\n' + '='*70)
    print('EVALUACIÓN COMPLETADA')
    print('='*70)


if __name__ == '__main__':
    main()
