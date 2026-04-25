"""Descarga parquets de la TLC (NYC Taxi & Limousine Commission) a data/raw/.

Ejemplos de uso:

    # Un mes concreto del dataset yellow (por defecto)
    python -m src.data.download_tlc --year-months 2025-01

    # Varios meses sueltos
    python -m src.data.download_tlc --year-months 2024-01 2024-06 2025-01

    # Año completo
    python -m src.data.download_tlc --years 2024

    # Combinar años y meses (producto cartesiano)
    python -m src.data.download_tlc --years 2023 2024 --months 1 2 3

    # Otro dataset
    python -m src.data.download_tlc --dataset green --years 2024 --months 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASETS = ("yellow", "green", "fhv", "fhvhv")


def download(dataset: str, year_month: str, force: bool = False) -> Path | None:
    filename = f"{dataset}_tripdata_{year_month}.parquet"
    url = f"{BASE_URL}/{filename}"
    dest = RAW_DIR / filename

    if dest.exists() and not force:
        print(f"[skip] ya existe: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")

    try:
        with urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            chunk = 1 << 20  # 1 MB
            with open(dest, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    written += len(buf)
                    if total:
                        pct = 100 * written / total
                        print(
                            f"  {written/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)",
                            end="\r",
                        )
            print()
    except HTTPError as e:
        # Limpiamos el fichero parcial si lo hubiera
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        print(f"[err ] {filename}: HTTP {e.code} — no disponible en el servidor")
        return None
    except URLError as e:
        print(f"[err ] {filename}: error de red — {e.reason}")
        return None

    print(f"[ok  ] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def expand_year_months(
    years: list[int] | None,
    months: list[int] | None,
    year_months: list[str] | None,
) -> list[str]:
    """Convierte las opciones del CLI en una lista ordenada y única de 'YYYY-MM'."""
    out: set[str] = set()

    if year_months:
        for ym in year_months:
            parse_year_month(ym)  # valida formato
            out.add(ym)

    if years:
        target_months = months if months else list(range(1, 13))
        for y in years:
            for m in target_months:
                out.add(f"{y:04d}-{m:02d}")

    return sorted(out)


def parse_year_month(value: str) -> str:
    try:
        year_str, month_str = value.split("-")
        year = int(year_str)
        month = int(month_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"'{value}' no es un YYYY-MM válido (ej: 2025-01)"
        ) from e
    if not (1 <= month <= 12):
        raise argparse.ArgumentTypeError(f"mes fuera de rango en '{value}'")
    if year < 2009:
        raise argparse.ArgumentTypeError(f"año demasiado antiguo en '{value}' (TLC empieza en 2009)")
    return f"{year:04d}-{month:02d}"


def parse_month(value: str) -> int:
    try:
        m = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"'{value}' no es un mes válido") from e
    if not (1 <= m <= 12):
        raise argparse.ArgumentTypeError(f"mes fuera de rango: {m}")
    return m


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Descarga parquets de viajes de la TLC de NYC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=DATASETS,
        default="yellow",
        help="Tipo de dataset (default: yellow)",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Uno o más años (ej: --years 2023 2024)",
    )
    p.add_argument(
        "--months",
        type=parse_month,
        nargs="+",
        help="Meses 1-12 (ej: --months 1 2 3). Se combinan con --years.",
    )
    p.add_argument(
        "--year-months",
        type=parse_year_month,
        nargs="+",
        help="Meses concretos en formato YYYY-MM (ej: --year-months 2024-01 2025-03)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Volver a descargar aunque el fichero ya exista en data/raw/",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    targets = expand_year_months(args.years, args.months, args.year_months)
    if not targets:
        print(
            "[err ] no has indicado qué descargar. Usa --year-months y/o --years (con --months opcional).",
            file=sys.stderr,
        )
        return 2

    print(f"[plan] dataset={args.dataset}  meses={len(targets)}  destino={RAW_DIR}")

    ok, fail = 0, 0
    for ym in targets:
        result = download(args.dataset, ym, force=args.force)
        if result is None:
            fail += 1
        else:
            ok += 1

    print(f"[done] ok={ok}  fallos={fail}  total={len(targets)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
