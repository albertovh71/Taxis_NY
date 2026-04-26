"""Tests detallados y análisis avanzado del modelo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from scipy import stats


def run_detailed_tests():
    """Ejecutar suite de tests detallados sobre el modelo."""
    print('='*70)
    print('SUITE DE TESTS DETALLADOS')
    print('='*70)

    # Cargar datos y modelo
    df = pd.read_parquet('data/processed/yellow_taxi_features.parquet')
    model = joblib.load('models/linear_model.joblib')

    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'hour_of_week',
        'is_holiday', 'is_holiday_eve',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_3h', 'rolling_mean_12h', 'rolling_mean_24h', 'rolling_mean_168h'
    ]
    target = 'trip_count'

    test_date = pd.to_datetime('2025-03-01')
    df_test = df[df['pickup_datetime'] >= test_date].copy()

    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test[target]
    y_pred = model.predict(X_test)

    # TEST 1: Distribución de errores
    print('\n' + '='*70)
    print('TEST 1: DISTRIBUCION DE ERRORES')
    print('='*70)

    errors = y_test.values - y_pred
    abs_errors = np.abs(errors)

    print(f'Errores absolutos:')
    print(f'  Media: {abs_errors.mean():.4f}')
    print(f'  Mediana: {np.median(abs_errors):.4f}')
    print(f'  Std: {abs_errors.std():.4f}')
    print(f'  Min: {abs_errors.min():.4f}')
    print(f'  Max: {abs_errors.max():.4f}')
    print(f'  P25: {np.percentile(abs_errors, 25):.4f}')
    print(f'  P75: {np.percentile(abs_errors, 75):.4f}')

    # TEST 2: Calibración
    print('\n' + '='*70)
    print('TEST 2: CALIBRACION')
    print('='*70)

    mean_real = y_test.mean()
    mean_pred = y_pred.mean()
    bias = mean_real - mean_pred

    print(f'Real (media): {mean_real:.4f}')
    print(f'Prediccion (media): {mean_pred:.4f}')
    print(f'Sesgo: {bias:.4f} ({100*bias/mean_real:.2f}%)')

    if abs(bias) < mean_real * 0.05:
        print('[PASS] Modelo bien calibrado (sesgo < 5%)')
    else:
        print('[FAIL] Modelo descalibrado (sesgo >= 5%)')

    # TEST 3: Percentiles de error
    print('\n' + '='*70)
    print('TEST 3: PERCENTILES DE ERROR')
    print('='*70)

    percentiles = [50, 75, 90, 95, 99]
    print(f'{"Percentil":<12} {"Error Abs":<15} {"% de casos"}')
    print('-'*40)

    for p in percentiles:
        threshold = np.percentile(abs_errors, p)
        pct_within = 100 * (abs_errors <= threshold).sum() / len(abs_errors)
        print(f'{p}:         {threshold:<15.4f} {pct_within:.1f}%')

    # TEST 4: Performance por deciles de demanda
    print('\n' + '='*70)
    print('TEST 4: PERFORMANCE POR DECILES DE DEMANDA')
    print('='*70)

    y_test_vals = y_test.values
    deciles = pd.qcut(y_test_vals, q=10, labels=False, duplicates='drop')

    print(f'{"Decil":<8} {"Min Dem":<12} {"Max Dem":<12} {"MAE":<10} {"MAPE"}')
    print('-'*55)

    for dec in sorted(np.unique(deciles)):
        mask = (deciles == dec)

        if mask.sum() == 0:
            continue

        min_dem = y_test_vals[mask].min()
        max_dem = y_test_vals[mask].max()
        mae = abs_errors[mask].mean()

        mask_nonzero = y_test_vals[mask] != 0
        if mask_nonzero.sum() > 0:
            mape = 100 * (abs_errors[mask][mask_nonzero] / y_test_vals[mask][mask_nonzero]).mean()
        else:
            mape = 0.0

        print(f'{int(dec):<8} {min_dem:<12.1f} {max_dem:<12.1f} {mae:<10.4f} {mape:.2f}%')

    # TEST 5: Tendencia temporal
    print('\n' + '='*70)
    print('TEST 5: TENDENCIA TEMPORAL (ERROR OVER TIME)')
    print('='*70)

    df_test['error'] = errors
    df_test['abs_error'] = abs_errors
    df_test['week'] = df_test['pickup_datetime'].dt.isocalendar().week

    weekly_mae = df_test.groupby('week')['abs_error'].mean()

    print('\nMAE por semana:')
    for week, mae in weekly_mae.items():
        print(f'  Semana {week}: {mae:.4f}')

    # TEST 6: Performance fin de semana
    print('\n' + '='*70)
    print('TEST 6: PERFORMANCE EN CONTEXTOS ESPECIALES')
    print('='*70)

    is_weekend = df_test['pickup_datetime'].dt.dayofweek >= 5

    weekday_mae = df_test[~is_weekend]['abs_error'].mean()
    weekend_mae = df_test[is_weekend]['abs_error'].mean()

    print(f'\nEntre semana (lunes-viernes):')
    print(f'  MAE: {weekday_mae:.4f}')
    print(f'  Muestras: {(~is_weekend).sum():,}')

    print(f'\nFin de semana (sabado-domingo):')
    print(f'  MAE: {weekend_mae:.4f}')
    print(f'  Muestras: {is_weekend.sum():,}')

    print(f'\nDiferencia: {abs(weekend_mae - weekday_mae):.4f} ({100*abs(weekend_mae - weekday_mae)/weekday_mae:.1f}%)')

    # TEST 7: Análisis de residuos (normalidad)
    print('\n' + '='*70)
    print('TEST 7: ANALISIS DE RESIDUOS')
    print('='*70)

    sample_size = min(5000, len(errors))
    sample_errors = np.random.choice(errors, size=sample_size, replace=False)

    stat, p_value = stats.shapiro(sample_errors)
    print(f'\nShapiro-Wilk Test (normalidad de residuos):')
    print(f'  Estadistico: {stat:.6f}')
    print(f'  P-value: {p_value:.6e}')

    if p_value > 0.05:
        print('  [PASS] Residuos normalmente distribuidos (p > 0.05)')
    else:
        print('  [WARN] Residuos NO normalmente distribuidos (p < 0.05)')

    # Autocorrelación
    print(f'\nAutocorrelacion de residuos:')
    acf_lag1 = pd.Series(errors).autocorr(lag=1)
    acf_lag24 = pd.Series(errors).autocorr(lag=24)

    print(f'  Lag 1: {acf_lag1:.4f}')
    print(f'  Lag 24: {acf_lag24:.4f}')

    if abs(acf_lag1) < 0.1:
        print('  [PASS] Baja autocorrelacion (lag 1)')
    else:
        print('  [WARN] Alta autocorrelacion (lag 1)')

    # TEST 8: Worst cases
    print('\n' + '='*70)
    print('TEST 8: PEORES CASOS')
    print('='*70)

    worst_idx = np.argsort(abs_errors)[-5:]
    print(f'\nTop 5 peores predicciones (mayor error absoluto):')
    print(f'{"Indice":<8} {"Real":<10} {"Prediccion":<12} {"Error Abs"}')
    print('-'*42)

    for idx in reversed(worst_idx):
        print(f'{idx:<8} {y_test.iloc[idx]:<10.2f} {y_pred[idx]:<12.2f} {abs_errors[idx]:<10.2f}')

    # Casos con predicción negativa
    negative_preds = (y_pred < 0).sum()
    print(f'\nPredicciones negativas: {negative_preds} ({100*negative_preds/len(y_pred):.3f}%)')

    if negative_preds > 0:
        print('  [WARN] El modelo puede predecir valores negativos')
    else:
        print('  [PASS] Ninguna prediccion negativa')

    # TEST 9: Estabilidad del modelo
    print('\n' + '='*70)
    print('TEST 9: ESTABILIDAD (REPRODUCIBILIDAD)')
    print('='*70)

    y_pred2 = model.predict(X_test)
    diff = np.abs(y_pred - y_pred2).max()

    print(f'Max diferencia entre dos predicciones: {diff:.10f}')

    if diff < 1e-10:
        print('[PASS] Modelo deterministico y reproducible')
    else:
        print('[FAIL] Modelo inestable')

    # TEST 10: Cobertura de features
    print('\n' + '='*70)
    print('TEST 10: COBERTURA DE FEATURES')
    print('='*70)

    print(f'\nNaN en features de entrada:')
    nan_found = False
    for col in feature_cols:
        nan_count = X_test[col].isna().sum()
        if nan_count > 0:
            print(f'  {col}: {nan_count} ({100*nan_count/len(X_test):.2f}%)')
            nan_found = True

    if not nan_found:
        print('  Ninguno - todas las features sin NaN')

    print('\n[OK] Tests completados')


if __name__ == '__main__':
    run_detailed_tests()
