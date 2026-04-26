# CLAUDE.md

Contexto persistente del proyecto para Claude Code. Se carga automáticamente en cada sesión.

## Resumen

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

## Decisiones de arquitectura

### Definidas

- **Granularidad temporal:** hora (en lugar de 15 min). Trade-off: menos ruido y más estable operacionalmente; las decisiones de despacho operan a escala horaria.
- **Datasets TLC:** Yellow + FHVHV combinados. Refleja mejor la realidad actual de NYC (Yellow tradicional + Uber/Lyft de volumen alto). Requiere estandarización de schemas en la limpieza.
- **Rango temporal:** 2023-2025 (3 años completos). Volumen mayor (~6.9M muestras hora-zona), mejor representación de patrones estacionales e impactos post-COVID.
- **Métrica principal:** MAPE por zona. Interpretable ("error 10%"), sensible a errores grandes, y refleja lo que importa operacionalmente en zonas de baja demanda.
- **Modelo seleccionado:** Linear Regression. MAE 0.26 en test (mejor que XGBoost). Rápido, interpretable, sin overfitting.

### Para fase 2 (después del baseline)

- **Fuentes externas:** clima (Meteostat/NOAA), festivos US, eventos NYC. Primero asegurar un baseline sólido solo con datos TLC; las externas son mejoras iterativas.

## Resultados del Modelo (Yellow + FHVHV 2023-2025)

**Dataset:** 6.9M muestras (3.4M Yellow + 2.9M FHVHV), 263 zonas, 3 años completos  
**Split:** Train 2023-2024 (4.6M), Test 2025 (1.9M)  
**Modelo:** Linear Regression

| Métrica | Train | Test | Status |
|---------|-------|------|--------|
| MAE | 0.6017 viajes | 0.2643 viajes | ✓ Test mejor |
| RMSE | 4.7591 viajes | 0.5410 viajes | ✓ Reducción 88% |
| MAPE | 3.01% | 5.66% | ⚠ +2.6 pp |
| R² | 0.9988 | 0.9999 | ✓ Excelente |

**Análisis por zona:**
- Mejores: zona 48 (MAPE 0.81%, demanda 134/h), zona 100 (MAPE 1%, demanda 78/h)
- Peores: zona 44 (MAPE 16.92%, demanda <0.01/h), zonas de baja demanda

**Análisis por hora:**
- Mejor rendimiento: 7:00-8:00 (MAPE ~5%), horas pico matutinas
- Peor rendimiento: 0:00-4:00 (MAPE ~6-7%), madrugada con menos volumen

## Decisiones pendientes

- Estrategia de validación (time-series split, walk-forward, etc.).
- Horizonte de predicción (1 hora adelante, 24 horas, etc.).
- Estrategia de deployment (batch diario, serving en tiempo real, etc.).
- Planes para fase 2 (fuentes externas, ajustes por zona especial).
