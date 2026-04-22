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

## Estado

En desarrollo inicial — estructura del proyecto creada.
