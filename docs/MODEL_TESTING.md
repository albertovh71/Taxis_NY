# Evaluación del Modelo de Predicción de Demanda de Taxis

## Resumen Ejecutivo

Se han desarrollado dos scripts comprehensivos para evaluar el modelo Linear Regression entrenado en datos de la TLC (2024-01 a 2025-03).

### Métricas Globales (Test Set: 188,181 muestras)

| Métrica | Valor | Interpretación |
|---------|-------|-----------------|
| **MAE** | 0.1245 viajes | Error promedio (bajo) |
| **RMSE** | 0.3942 viajes | Sensibilidad a outliers |
| **MAPE** | 2.24% | Error porcentual (excelente) |
| **R²** | 0.9999 | Varianza explicada (muy alta) |

**Conclusión:** El modelo tiene un desempeño **muy bueno** con un error medio de solo 2.24% en predicciones del test set.

---

## Scripts Disponibles

### 1. `src/models/evaluate_model.py`

Evaluación comprehensiva en tres períodos temporales:

```bash
python -m src.models.evaluate_model
```

**Qué genera:**

- Comparación train vs test (gráficos de scatter y residuos)
- Análisis por zona (MAPE, MAE para cada zona de NYC)
- Análisis por hora del día (MAPE horario: 1.45% - 7.60%)
- Gráficos:
  - `reports/model_predictions_scatter.png` — Actual vs Predicción
  - `reports/model_mape_by_hour.png` — Error por hora
  - `reports/model_residuals_distribution.png` — Distribución de residuos

**Archivos salida:**

- `reports/model_metrics_global.csv` — Métricas generales (train/test)
- `reports/model_metrics_by_zone.csv` — MAPE, MAE por zona
- `reports/model_metrics_by_hour.csv` — Métricas por hora del día
- `reports/model_predictions.parquet` — Predicciones completas con errores

---

### 2. `src/models/detailed_tests.py`

Suite de 10 tests detallados para validar robustez:

```bash
python -m src.models.detailed_tests
```

**Tests incluidos:**

1. **Distribución de errores** — Percentiles, mediana, std
2. **Calibración** — Sesgo entre real y predicción (0.04% ✓)
3. **Percentiles de error** — P50, P75, P90, P95, P99
4. **Performance por déciles de demanda** — Precisión en demanda baja vs alta
5. **Tendencia temporal** — MAE por semana (estable, salvo semana 14)
6. **Contextos especiales** — Fin de semana vs entre semana (3% de diferencia)
7. **Análisis de residuos** — Normalidad (Shapiro-Wilk) y autocorrelación
8. **Peores casos** — Top 5 predicciones con mayor error
9. **Estabilidad** — Reproducibilidad del modelo (perfecta)
10. **Cobertura de features** — Verificación de NaN

---

## Hallazgos Clave

### ✓ Puntos Fuertes

1. **Excelente calibración:** Sesgo de solo 0.04% (modelo predice valores correctos en promedio)
2. **Baja variabilidad:** El 95% de los errores están por debajo de 0.67 viajes
3. **Performance en horas pico:** MAPE 1.45% - 1.72% en 8:00 - 21:00
4. **Estabilidad:** Reproducible en todas las ejecuciones
5. **Sin NaN en features:** Todas las features procesadas correctamente

### ⚠ Advertencias

1. **Predicciones negativas:** 14.5% de las predicciones son negativas
   - **Causa:** Linear Regression no está limitado a [0, ∞)
   - **Solución:** Aplicar `y_pred = np.maximum(y_pred, 0)` en producción

2. **Peor rendimiento en madrugada:** MAPE 6-7% en horas 1-4
   - **Causa:** Menor volumen y más volatilidad en esas horas
   - **Mitigation:** Considerar modelos separados por franja horaria

3. **Autocorrelación en residuos:** Lag 1 = 0.1982
   - **Implicación:** Hay dependencia temporal no completamente capturada
   - **Mejora:** Incluir más features lagged o usar ARIMA/estado-espacio

4. **Residuos no normales:** Shapiro-Wilk p-value = 8.08e-82
   - **Esperado:** Los conteos tienen distribución Poisson, no Normal
   - **No crítico:** Linear Regression sigue siendo válido por Teorema Central del Límite

### Variación por Zona

- **Mejores zonas:** Demanda = 0 → MAPE 0% (trivial)
- **Peores zonas:** Zona 138 MAPE 4.76%, Zona 236 MAPE 4.16%
- **Promedio:** MAPE global de 2.24% es muy competitivo

### Variación Temporal

| Período | MAE | Observación |
|---------|-----|------------|
| Entre semana | 0.1233 | Predecible |
| Fin de semana | 0.1270 | +3% error (más variabilidad) |
| Semana 14 | 0.2276 | Pico anómalo (datos incompletos?) |

---

## Recomendaciones

### Corto Plazo (Producción)

1. **Clampear predicciones:**
   ```python
   y_pred = np.maximum(y_pred, 0)
   ```

2. **Monitorear en tiempo real:**
   - Comparar predicciones vs realizaciones diarias
   - Alertar si MAPE diaria > 5%

3. **Usar por zonas de demanda:**
   - Separar modelo para zonas de baja demanda (< 10 viajes/hora)

### Mediano Plazo (Mejora del Modelo)

1. **Considerar Poisson o Negative Binomial Regression:**
   - Respeta la naturaleza de conteos
   - Evita predicciones negativas naturalmente

2. **Agregar features exógenas (Fase 2):**
   - Clima (temperatura, precipitación)
   - Eventos (conciertos, deportes)
   - Festivos e holidays

3. **Aumentar granularidad temporal:**
   - Actualmente: 1 hora
   - Considerar: 30 min (si el costo computacional lo permite)

4. **Ensembles:**
   - Combinar Linear Regression + XGBoost
   - Pesar por confianza (zonas de baja demanda → más peso a promedio)

### Validación Continua

- Ejecutar `evaluate_model.py` mensualmente
- Comparar MAPE de este mes vs baseline (2.24%)
- Si degrada > 10%, reentrenar

---

## Estructura de Archivos

```
reports/
├── model_metrics_global.csv         # Métricas train/test
├── model_metrics_by_zone.csv        # Performance por zona
├── model_metrics_by_hour.csv        # Performance por hora
├── model_predictions.parquet        # Predicciones detalladas
├── model_predictions_scatter.png    # Gráfico actual vs pred
├── model_mape_by_hour.png           # MAPE por hora
└── model_residuals_distribution.png # Hist. de residuos
```

---

## Cómo Interpretar los Resultados

### Ejemplo: Predicción en Zona 10, 14:00

```
Real: 45 viajes
Predicción: 44.8 viajes
Error: -0.2 viajes
Error %: -0.44%
```

Este es un resultado típico (mediana de error es solo 0.0178 viajes).

### Ejemplo: Peor Caso (Índice 56432)

```
Real: 0 viajes
Predicción: 18.41 viajes
Error: 18.41 viajes (100% error)
```

- Esto ocurre cuando la zona realmente tiene 0 trips pero el modelo predice demanda
- Es raro (estadísticamente esperado dado el tamaño del dataset)
- Solución: Clampear a 0 en producción

---

## Próximos Pasos

1. **Guardar las métricas actuales como baseline** en version control
2. **Implementar monitoreo en producción** (si se despliega)
3. **Diseñar estrategia de re-training** (mensual? trimestral?)
4. **Evaluar trade-offs** de complejidad vs precisión si consideramos modelos más avanzados
