"""Procesar y combinar datos Yellow + FHVHV."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from glob import glob


def process_yellow_data(parquet_files: list[str]) -> pd.DataFrame:
    """Procesar datos de Yellow Taxis - procesar y agregar por archivo."""
    print(f'[yellow] procesando {len(parquet_files)} archivos (procesando individual)...')

    all_agg = []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=[
                'tpep_pickup_datetime',
                'PULocationID',
                'trip_distance',
                'passenger_count'
            ])
            df.columns = ['pickup_datetime', 'PULocationID', 'trip_distance', 'passenger_count']
            df['dataset'] = 'yellow'

            # Agregar inmediatamente por zone-hora
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['hour_start'] = df['pickup_datetime'].dt.floor('H')

            agg_file = df.groupby(['hour_start', 'PULocationID']).agg({
                'trip_distance': 'sum',
                'passenger_count': 'sum'
            }).reset_index()

            trip_count = df.groupby(['hour_start', 'PULocationID']).size().reset_index(name='trip_count')
            agg_file = agg_file.merge(trip_count, on=['hour_start', 'PULocationID'])
            agg_file.columns = ['pickup_datetime', 'PULocationID', 'total_distance', 'total_passengers', 'trip_count']
            agg_file['dataset'] = 'yellow'

            all_agg.append(agg_file)
            print(f'  [ok] {Path(pf).name}: {len(agg_file):,} registros')

        except Exception as e:
            print(f'  [warn] error en {Path(pf).name}: {e}')
            continue

    if not all_agg:
        print('[error] no hay datos de Yellow')
        return pd.DataFrame()

    df = pd.concat(all_agg, ignore_index=True)
    print(f'[yellow] total {len(df):,} registros hora-zona')
    return df


def process_fhvhv_data(parquet_files: list[str]) -> pd.DataFrame:
    """Procesar datos de FHVHV (Uber/Lyft) - procesar y agregar por archivo."""
    print(f'[fhvhv] procesando {len(parquet_files)} archivos (procesando individual)...')

    all_agg = []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=[
                'pickup_datetime',
                'PULocationID',
            ])
            df['trip_distance'] = 0
            df['passenger_count'] = 1
            df['dataset'] = 'fhvhv'

            # Agregar inmediatamente por zone-hora
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['hour_start'] = df['pickup_datetime'].dt.floor('H')

            agg_file = df.groupby(['hour_start', 'PULocationID']).agg({
                'trip_distance': 'sum',
                'passenger_count': 'sum'
            }).reset_index()

            trip_count = df.groupby(['hour_start', 'PULocationID']).size().reset_index(name='trip_count')
            agg_file = agg_file.merge(trip_count, on=['hour_start', 'PULocationID'])
            agg_file.columns = ['pickup_datetime', 'PULocationID', 'total_distance', 'total_passengers', 'trip_count']
            agg_file['dataset'] = 'fhvhv'

            all_agg.append(agg_file)
            print(f'  [ok] {Path(pf).name}: {len(agg_file):,} registros')

        except Exception as e:
            print(f'  [warn] error en {Path(pf).name}: {e}')
            continue

    if not all_agg:
        print('[warn] no hay datos de FHVHV')
        return pd.DataFrame()

    df = pd.concat(all_agg, ignore_index=True)
    print(f'[fhvhv] total {len(df):,} registros hora-zona')
    return df


def aggregate_combined_data(dfs_to_combine: list[pd.DataFrame]) -> pd.DataFrame:
    """Combinar y agregar datos de múltiples fuentes."""
    print('[aggregate] combinando datos de Yellow y FHVHV...')

    df = pd.concat(dfs_to_combine, ignore_index=True)

    # Agrupar por zona y hora en caso de duplicados
    agg = df.groupby(['pickup_datetime', 'PULocationID']).agg({
        'trip_count': 'sum',
        'total_distance': 'sum',
        'total_passengers': 'sum'
    }).reset_index()

    print(f'[aggregate] {len(agg):,} registros hora-zona combinados')

    return agg


def main():
    """Procesar datos combinados."""
    print('='*70)
    print('PROCESAMIENTO: YELLOW + FHVHV (2023-2025)')
    print('='*70)

    # Buscar archivos
    print('\n[search] buscando archivos parquet descargados...')

    yellow_files = sorted(glob('data/raw/yellow_tripdata_*.parquet'))
    fhvhv_files = sorted(glob('data/raw/fhvhv_tripdata_*.parquet'))

    print(f'  Yellow: {len(yellow_files)} archivos')
    print(f'  FHVHV: {len(fhvhv_files)} archivos')

    if not yellow_files and not fhvhv_files:
        print('[error] no hay archivos parquet en data/raw/')
        return

    # Procesar
    print('\n[process] cargando y procesando datos...')

    dfs_to_combine = []

    if yellow_files:
        df_yellow = process_yellow_data(yellow_files)
        if not df_yellow.empty:
            dfs_to_combine.append(df_yellow)

    if fhvhv_files:
        df_fhvhv = process_fhvhv_data(fhvhv_files)
        if not df_fhvhv.empty:
            dfs_to_combine.append(df_fhvhv)

    if not dfs_to_combine:
        print('[error] no se cargaron datos')
        return

    # Combinar
    print('\n[combine] combinando datos...')
    df_agg = aggregate_combined_data(dfs_to_combine)
    print(f'  Total: {len(df_agg):,} registros')
    print(f'  Rango: {df_agg["pickup_datetime"].min()} a {df_agg["pickup_datetime"].max()}')
    print(f'  Zonas únicas: {df_agg["PULocationID"].nunique()}')

    # Guardar datos procesados
    print('\n[save] guardando datos procesados...')

    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'yellow_fhvhv_by_zone_hour.parquet'
    df_agg.to_parquet(output_file, index=False)
    print(f'  {output_file}')

    # Estadísticas
    print('\n[stats] estadísticas del dataset procesado:')
    print(f'  Filas: {len(df_agg):,}')
    print(f'  Viajes totales: {df_agg["trip_count"].sum():,}')
    print(f'  Viajes por hora-zona (promedio): {df_agg["trip_count"].mean():.2f}')
    print(f'  Viajes por hora-zona (mediana): {df_agg["trip_count"].median():.2f}')
    print(f'  Viajes por hora-zona (max): {df_agg["trip_count"].max():.0f}')

    print('\n[done] procesamiento completado')


if __name__ == '__main__':
    main()
