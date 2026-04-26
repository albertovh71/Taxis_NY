# CLAUDE.md

Contexto persistente del proyecto para Claude Code. Se carga automáticamente en cada sesión.

## Resumen AVi ha estado aqui

Proyecto de Machine Learning para **predecir la demanda de taxis en Nueva York** por zona y franja horaria, usando datos históricos de la TLC (Taxi & Limousine Commission) y, más adelante, fuentes externas (clima, festivos, eventos).

Estado: desarrollo inicial. Estructura del proyecto creada y script de ingesta operativo.

## Idioma y comunicación

- **Comunicación con el usuario:** español.
- **Código:** nombres de variables, funciones y módulos **en inglés** (snake_case).
- **Comentarios y docstrings:** en español.
- **Mensajes de log y output del usuario** (prints, errores, README): en español.
- **Mensajes de commit:** en español, con prefijo convencional (`feat`, `fix`, `chore`, `docs`, `refactor`, `test`).

## Stack

- **Lenguaje:** Python 3.11+
- **Manipulación de datos:** pandas (+ pyarrow para leer parquets).
- **ML:** scikit-learn como base. Otros frameworks (XGBoost, LightGBM, etc.) se valorarán al llegar al modelado.
- **Notebooks:** Jupyter para EDA y reportes.
- **Visualización:** matplotlib / seaborn por defecto; plotly o folium si necesitamos mapas.
- **Formato y linting:**
  - **black** — formateador automático (estilo único, sin discusión).
  - **ruff** — linter rápido (errores, imports sin usar, antipatrones, orden de imports).
  - Configuración en `pyproject.toml` (línea 100, target Python 3.11).

## Estructura del proyecto

```
Taxis_NY/
├── configs/          # Configuración de experimentos (YAML/JSON)
├── data/
│   ├── raw/          # Datos originales sin modificar (TLC parquets)
│   ├── interim/      # Datos en transformación intermedia
│   ├── processed/    # Datasets listos para entrenamiento
│   └── external/     # Fuentes externas (clima, festivos, etc.)
├── docs/             # Documentación adicional
├── models/           # Modelos entrenados serializados
├── notebooks/
│   ├── exploratory/  # EDA y análisis inicial
│   └── reports/      # Notebooks pulidos para presentar
├── reports/figures/  # Gráficos exportados
├── src/
│   ├── data/         # Ingesta y limpieza
│   ├── features/     # Feature engineering
│   ├── models/       # Entrenamiento y evaluación
│   └── visualization/# Gráficos y mapas
└── tests/            # Tests unitarios
```

**Regla importante:** los datos crudos (`data/raw/`, `data/interim/`, `data/processed/`, `data/external/`) y los modelos entrenados (`models/`) **no se commitean** — están en `.gitignore`. Solo se mantiene `.gitkeep` para preservar la estructura.

## Comandos habituales

### Ingesta de datos TLC

Descarga parquets mensuales a `data/raw/`. CLI con varias formas de seleccionar meses:

```bash
# Un mes concreto del dataset yellow (por defecto)
python -m src.data.download_tlc --year-months 2025-01

# Varios meses sueltos
python -m src.data.download_tlc --year-months 2024-01 2024-06 2025-01

# Año completo
python -m src.data.download_tlc --years 2024

# Producto cartesiano años × meses
python -m src.data.download_tlc --years 2023 2024 --months 1 2 3

# Otro dataset (yellow | green | fhv | fhvhv)
python -m src.data.download_tlc --dataset green --years 2024 --months 1
```

Opción `--force` para redescargar aunque el fichero exista.

### Formato y linting

```bash
# Formatear todo el código (modifica ficheros)
black src tests

# Comprobar formato sin modificar
black --check src tests

# Lint (detectar problemas)
ruff check src tests

# Lint + arreglar lo que se pueda automáticamente
ruff check --fix src tests
```

## Convenciones de código

- **Estilo:** PEP 8, indentación con 4 espacios.
- **Nombres:** `snake_case` para variables/funciones, `PascalCase` para clases, `UPPER_SNAKE` para constantes.
- **Type hints:** úsalos donde aporten claridad (firmas de funciones públicas, returns no triviales). No es obligatorio en cada variable local.
- **Docstrings:** en español, breves. Una línea para funciones simples; formato más completo solo cuando el comportamiento sea no obvio.
- **Imports:** orden estándar (stdlib → terceros → locales), separados por línea en blanco.
- **Sin comentarios obvios:** el comentario explica el *porqué* cuando no es evidente, no el *qué*.

## Dominio TLC (notas)

La TLC publica varios datasets de viajes en `https://d37ci6vzurychx.cloudfront.net/trip-data/`:

- **`yellow_tripdata_YYYY-MM.parquet`** — Yellow Taxis (Manhattan principalmente, hail en calle). Disponibles desde 2009.
- **`green_tripdata_YYYY-MM.parquet`** — Green Taxis (boroughs externos). Desde 2013.
- **`fhv_tripdata_YYYY-MM.parquet`** — For-Hire Vehicles tradicionales (limusinas, coches con base).
- **`fhvhv_tripdata_YYYY-MM.parquet`** — High-Volume FHV (Uber, Lyft, Via, Juno). Desde 2019.

**Columnas clave** (yellow/green): `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`, `DOLocationID`, `passenger_count`, `trip_distance`, `fare_amount`, `total_amount`.

**Zonas:** las posiciones `PULocationID`/`DOLocationID` referencian las **TaxiZones** de la TLC (~265 zonas). El shapefile y un CSV de lookup están publicados por la TLC. Será probablemente la **granularidad espacial** del problema de predicción.

**Granularidad temporal:** decisión pendiente (hora, 30 min, 15 min). Trade-off entre detalle y volumen.

## Workflow de git

- Rama principal: `main`.
- Commits pequeños y enfocados, mensaje en español con prefijo convencional.
- Antes de commitear cualquier cosa que toque `data/` o `models/`, comprobar que `.gitignore` lo excluye correctamente.
- No commitear `.claude/` (configuración local de Claude Code).

## Decisiones pendientes

- Granularidad temporal del target (hora vs 15 min).
- Conjunto de datasets TLC a usar (solo yellow, o combinar con fhvhv).
- Rango temporal de entrenamiento.
- Fuentes externas a integrar (clima vía Meteostat/NOAA, festivos US, eventos NYC).
- Métrica principal de evaluación (MAE, RMSE, MAPE por zona).
