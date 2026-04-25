"""Agregación de datos por zona y hora."""
from __future__ import annotations

import pandas as pd


def aggregate_by_zone_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Agregar viajes por zona pickup (PULocationID) y hora.

    Retorna un dataframe con:
    - PULocationID: zona
    - pickup_datetime: timestamp a la hora (inicio de la hora)
    - trip_count: número de viajes
    - avg_fare: tarifa promedio
    - avg_trip_distance: distancia promedio
    - avg_passenger_count: pasajeros promedio
    """
    # Agrupar por zona y hora
    df = df.copy()

    # Redondear al inicio de la hora
    df['pickup_datetime'] = df['tpep_pickup_datetime'].dt.floor('h')

    # Agregación
    agg_dict = {
        'VendorID': 'count',  # trip_count
        'fare_amount': 'mean',
        'trip_distance': 'mean',
        'passenger_count': 'mean',
    }

    agg = (
        df.groupby(['PULocationID', 'pickup_datetime'])
        .agg(agg_dict)
        .rename(columns={'VendorID': 'trip_count'})
        .reset_index()
    )

    # Redondear valores
    agg['avg_fare'] = agg['fare_amount'].round(2)
    agg['avg_trip_distance'] = agg['trip_distance'].round(2)
    agg['avg_passenger_count'] = agg['passenger_count'].round(2)

    # Dropear columnas sin renombrar
    agg = agg[['PULocationID', 'pickup_datetime', 'trip_count', 'avg_fare', 'avg_trip_distance', 'avg_passenger_count']]

    print(f"[agg] registros agregados: {len(agg):,} (zona-hora combos)")
    print(f"[agg] rango de fechas: {agg['pickup_datetime'].min()} a {agg['pickup_datetime'].max()}")
    print(f"[agg] zonas únicas: {agg['PULocationID'].nunique()}")

    return agg
