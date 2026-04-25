"""Feature engineering para predicción de demanda de taxis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import holidays as holidays_lib


def build_features(
    input_path: str = 'data/processed/yellow_taxi_by_zone_hour.parquet',
    output_path: str = 'data/processed/yellow_taxi_features.parquet',
    date_from: str = '2024-01-01',
    date_to: str = '2025-03-31',
) -> pd.DataFrame:
    """Construir features para modelo de predicción de demanda.

    Pasos:
    1. Leer dataset procesado
    2. Filtrar por rango de fechas válidas
    3. Completar grid zona-hora (reindex con 0s)
    4. Features temporales
    5. Festivos US
    6. Lags por zona
    7. Rolling averages por zona
    8. Guardar resultado
    """
    print('[features] iniciando construcción de features')

    # Leer
    print(f'[read] leyendo {input_path}')
    df = pd.read_parquet(input_path)
    print(f'  shape inicial: {df.shape}')

    # Convertir a datetime si no lo es
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Paso 1: Filtrar fechas
    print(f'[filter] filtrando a rango {date_from} a {date_to}')
    date_from_dt = pd.to_datetime(date_from)
    date_to_dt = pd.to_datetime(date_to)
    df = df[(df['pickup_datetime'] >= date_from_dt) & (df['pickup_datetime'] <= date_to_dt)].copy()
    print(f'  shape después filtro: {df.shape}')

    # Paso 2: Completar grid zona-hora
    print('[grid] completando grid zona-hora')
    # Si hay duplicados (zona, hora), agrupar y sumar trip_count (agregar registros duplicados)
    df = df.groupby(['PULocationID', 'pickup_datetime'], as_index=False).agg({
        'trip_count': 'sum',
        'avg_fare': 'mean',
        'avg_trip_distance': 'mean',
        'avg_passenger_count': 'mean',
    })
    print(f'  shape después consolidación: {df.shape}')

    zones = sorted(df['PULocationID'].unique())
    dates = pd.date_range(date_from_dt, date_to_dt, freq='h')
    index = pd.MultiIndex.from_product([zones, dates], names=['PULocationID', 'pickup_datetime'])

    df_full = df.set_index(['PULocationID', 'pickup_datetime']).reindex(index, fill_value=0).reset_index()
    print(f'  shape después reindex: {df_full.shape}')
    print(f'  zonas: {len(zones)}, horas: {len(dates)}')

    # Paso 3: Features temporales
    print('[temporal] añadiendo features temporales')
    df_full['hour'] = df_full['pickup_datetime'].dt.hour
    df_full['day_of_week'] = df_full['pickup_datetime'].dt.dayofweek  # lunes=0
    df_full['day_of_month'] = df_full['pickup_datetime'].dt.day
    df_full['month'] = df_full['pickup_datetime'].dt.month
    df_full['is_weekend'] = (df_full['day_of_week'] >= 5).astype(int)
    df_full['hour_of_week'] = df_full['day_of_week'] * 24 + df_full['hour']

    # Paso 4: Festivos US
    print('[holidays] añadiendo features de festivos')
    us_holidays = holidays_lib.US(years=range(2024, 2026))
    df_full['is_holiday'] = df_full['pickup_datetime'].dt.date.apply(lambda x: int(x in us_holidays))

    # is_holiday_eve: día anterior a un festivo
    next_day = df_full['pickup_datetime'] + pd.Timedelta(days=1)
    df_full['is_holiday_eve'] = next_day.dt.date.apply(lambda x: int(x in us_holidays)).astype(int)

    # Paso 5: Lags por zona
    print('[lags] calculando lags por zona')
    df_full = df_full.sort_values(['PULocationID', 'pickup_datetime']).reset_index(drop=True)

    for lag_hours in [1, 2, 3, 24, 168]:
        col_name = f'lag_{lag_hours}h'
        df_full[col_name] = df_full.groupby('PULocationID')['trip_count'].shift(lag_hours)

    # Paso 6: Rolling averages por zona
    print('[rolling] calculando promedios móviles por zona')
    for window_hours in [3, 12, 24, 168]:
        col_name = f'rolling_mean_{window_hours}h'
        df_full[col_name] = (
            df_full.groupby('PULocationID')['trip_count']
            .rolling(window=window_hours, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Paso 7: Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'[save] guardando a {output_path}')
    df_full.to_parquet(output_path, index=False)

    # Resumen
    print(f'\n[summary] shape final: {df_full.shape}')
    print(f'[summary] columnas: {list(df_full.columns)}')
    print(f'[summary] NaNs por columna:')
    for col in df_full.columns:
        nan_count = df_full[col].isna().sum()
        if nan_count > 0:
            pct = 100 * nan_count / len(df_full)
            print(f'  {col}: {nan_count:,} ({pct:.2f}%)')

    return df_full


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = 'data/processed/yellow_taxi_by_zone_hour.parquet'

    df_features = build_features(input_path=input_path)
    print(f'\nPrimeras filas:')
    print(df_features.head(3))
