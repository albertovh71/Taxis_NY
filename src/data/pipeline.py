"""Pipeline: ingesta → limpieza → agregación."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.clean import clean_yellow_taxi
from src.data.aggregate import aggregate_by_zone_hour


def run_pipeline(
    months: list[str] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Ejecutar pipeline completo: limpiar y agregar datos de TLC.

    Args:
        months: lista de YYYY-MM a procesar. Si None, procesa todos los que encuentre.
        output_path: ruta donde guardar el resultado parquet.
                    Si None, no guarda (solo retorna dataframe).

    Returns:
        DataFrame agregado por zona-hora.
    """
    raw_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw'

    # Encontrar parquets
    if months is None:
        parquets = sorted(raw_dir.glob('yellow_tripdata_*.parquet'))
        months = [p.stem.replace('yellow_tripdata_', '') for p in parquets]
    else:
        parquets = [raw_dir / f'yellow_tripdata_{m}.parquet' for m in months]

    if not parquets:
        raise ValueError(f'No parquets encontrados en {raw_dir}')

    print(f'[pipeline] procesando {len(parquets)} meses: {months}')

    # Leer y concatenar
    dfs = []
    for parquet in parquets:
        if not parquet.exists():
            print(f'[skip] {parquet.name} no existe')
            continue
        print(f'[read] {parquet.name}', end=' ... ')
        df = pd.read_parquet(parquet)
        dfs.append(df)
        print(f'{len(df):,} filas')

    if not dfs:
        raise ValueError('Sin datos para procesar')

    # Procesar por lotes (limpiar y agregar cada mes por separado, luego concatenar)
    print('\n[processing lotes]')
    agg_list = []
    for df in dfs:
        print(f'  limpiando y agregando {len(df):,} filas...', end=' ')
        df_clean = clean_yellow_taxi(df)
        df_agg_batch = aggregate_by_zone_hour(df_clean)
        agg_list.append(df_agg_batch)
        print(f'-> {len(df_agg_batch):,} zona-hora combos')

    # Concatenar resultados agregados
    df_agg = pd.concat(agg_list, ignore_index=True)
    print(f'\n[concat] total zona-hora registros: {len(df_agg):,}')

    # Guardar
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_parquet(output_path, index=False)
        print(f'\n[save] guardado a {output_path}')

    return df_agg


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        months = sys.argv[1].split(',')
    else:
        months = None

    df = run_pipeline(
        months=months,
        output_path='data/processed/yellow_taxi_by_zone_hour.parquet',
    )

    print(f'\nResultado final: {len(df):,} registros (zona-hora)')
    print(df.head())
