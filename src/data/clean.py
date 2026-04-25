"""Limpieza de datos de la TLC."""
from __future__ import annotations

import pandas as pd


def clean_yellow_taxi(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar dataset de Yellow Taxi removiendo registros sospechosos.

    Criterios de filtrado:
    - Fare > 0 (elimina viajes gratis/negativos)
    - trip_distance > 0 (elimina viajes sin desplazamiento)
    - trip_duration >= 1 min (elimina registros instantáneos)
    - PULocationID y DOLocationID válidos (1-265)
    - passenger_count > 0 (elimina viajes sin pasajeros)
    """
    initial_count = len(df)

    # Calcular duración si no existe
    if 'trip_duration_min' not in df.columns:
        df = df.copy()
        df['trip_duration_min'] = (
            (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])
            .dt.total_seconds() / 60
        )

    # Aplicar filtros
    mask = (
        (df['fare_amount'] > 0) &
        (df['trip_distance'] > 0) &
        (df['trip_duration_min'] >= 1) &
        (df['PULocationID'] >= 1) &
        (df['PULocationID'] <= 265) &
        (df['DOLocationID'] >= 1) &
        (df['DOLocationID'] <= 265) &
        (df['passenger_count'] > 0)
    )

    df_clean = df[mask].copy()

    removed = initial_count - len(df_clean)
    pct = 100 * removed / initial_count

    print(f"[clean] registros iniciales: {initial_count:,}")
    print(f"[clean] registros removidos: {removed:,} ({pct:.1f}%)")
    print(f"[clean] registros finales: {len(df_clean):,}")

    return df_clean
