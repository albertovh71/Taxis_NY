# Taxis_NY

Proyecto de Machine Learning para la predicción de la demanda de taxis en Nueva York.

## Objetivo

Entrenar modelos de ML capaces de predecir la demanda de viajes en taxi en distintas zonas y franjas horarias de la ciudad de Nueva York, utilizando datos históricos de la TLC (Taxi & Limousine Commission) y fuentes externas relevantes (clima, festivos, eventos, etc.).

## Estructura del proyecto

```
Taxis_NY/
├── configs/          # Configuración de experimentos (YAML/JSON)
├── data/
│   ├── raw/          # Datos originales sin modificar
│   ├── interim/      # Datos en transformación intermedia
│   ├── processed/    # Datasets listos para entrenamiento
│   └── external/     # Fuentes externas (clima, festivos, etc.)
├── docs/             # Documentación adicional
├── models/           # Modelos entrenados serializados
├── notebooks/
│   ├── exploratory/  # EDA y análisis inicial
│   └── reports/      # Notebooks pulidos
├── reports/
│   └── figures/      # Gráficos y visualizaciones
├── src/
│   ├── data/         # Ingesta y limpieza
│   ├── features/     # Feature engineering
│   ├── models/       # Entrenamiento y evaluación
│   └── visualization/# Gráficos y mapas
└── tests/            # Tests unitarios
```

## Ingesta de datos

Script para descargar parquets mensuales de la TLC a `data/raw/`:

```bash
# Un mes concreto del dataset yellow (por defecto)
python -m src.data.download_tlc --year-months 2025-01

# Varios meses sueltos
python -m src.data.download_tlc --year-months 2024-01 2024-06 2025-01

# Año completo
python -m src.data.download_tlc --years 2024

# Combinar años y meses (producto cartesiano)
python -m src.data.download_tlc --years 2023 2024 --months 1 2 3

# Otro dataset (yellow | green | fhv | fhvhv)
python -m src.data.download_tlc --dataset green --years 2024 --months 1
```

Opciones extra: `--force` para redescargar aunque el fichero ya exista. Si un mes no está disponible en el servidor, el script lo reporta y continúa con el resto.

## Version 1.0.0 - ✅ COMPLETADA

**Modelo entrenado y listo para uso**

### Incluye:
- ✅ Modelo Linear Regression entrenado (Yellow + FHVHV 2023-2025)
- ✅ App Streamlit con selector de fecha/zona/hora
- ✅ Comparación real vs predicción para datos históricos
- ✅ Scripts de evaluación y testing
- ✅ Pipeline automatizado de ingesta → procesamiento → entrenamiento

### Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| Dataset | Yellow + FHVHV (2023-2025) |
| Muestras | 6.9M registros hora-zona |
| MAE Test | 0.264 viajes/hora |
| RMSE Test | 0.541 viajes/hora |
| MAPE Test | 5.66% |
| R² | 0.9999 |

### Uso Rápido

```bash
# Ejecutar app web
streamlit run app_demo.py

# Acceder a http://localhost:8501
```

## Estado Anterior

En desarrollo inicial — estructura del proyecto e ingesta de datos TLC.
