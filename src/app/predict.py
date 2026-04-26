"""Lógica de predicción para fechas futuras."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def get_features_for_datetime(
    dt: pd.Timestamp,
    zone_id: int,
    history_df: pd.DataFrame,
) -> dict:
    """Calcular features para una zona y datetime específicos.

    Usa el histórico para los lags, calcula features temporales.
    """
    features = {}

    # Temporales
    features["hour"] = dt.hour
    features["day_of_week"] = dt.dayofweek
    features["day_of_month"] = dt.day
    features["month"] = dt.month
    features["is_weekend"] = int(dt.dayofweek >= 5)
    features["hour_of_week"] = dt.dayofweek * 24 + dt.hour

    # Festivos US
    us_holidays = holidays.US(years=dt.year)
    features["is_holiday"] = int(dt.date() in us_holidays)
    next_day = dt + pd.Timedelta(days=1)
    features["is_holiday_eve"] = int(next_day.date() in us_holidays)

    # Lags desde histórico
    zone_history = history_df[history_df["PULocationID"] == zone_id].sort_values("pickup_datetime")

    def get_lag_value(lag_hours: int) -> float:
        """Obtener valor de lag desde el histórico."""
        target_time = dt - pd.Timedelta(hours=lag_hours)
        match = zone_history[zone_history["pickup_datetime"] == target_time]
        if len(match) > 0:
            return match.iloc[0]["trip_count"]
        # Fallback: media de la misma hora del día en el histórico
        if len(zone_history) > 0:
            same_hour = zone_history[zone_history["pickup_datetime"].dt.hour == target_time.hour]
            if len(same_hour) > 0:
                return same_hour["trip_count"].mean()
        return 0.0

    features["lag_1h"] = get_lag_value(1)
    features["lag_2h"] = get_lag_value(2)
    features["lag_3h"] = get_lag_value(3)
    features["lag_24h"] = get_lag_value(24)
    features["lag_168h"] = get_lag_value(168)

    # Rolling averages: usar media histórica de la zona
    zone_mean = zone_history["trip_count"].mean()
    features["rolling_mean_3h"] = zone_mean
    features["rolling_mean_12h"] = zone_mean
    features["rolling_mean_24h"] = zone_mean
    features["rolling_mean_168h"] = zone_mean

    return features


def predict_demand(
    zone_id: int,
    target_datetime: pd.Timestamp,
    history_df: pd.DataFrame,
    model,
) -> float:
    """Predecir demanda (viajes/hora) para una zona y datetime.

    Args:
        zone_id: LocationID (1-265)
        target_datetime: datetime futuro
        history_df: histórico reciente para calcular lags
        model: modelo LinearRegression entrenado

    Returns:
        Número de viajes predichos (clipeado a 0)
    """
    features = get_features_for_datetime(target_datetime, zone_id, history_df)

    # Orden de features (debe coincidir con el entrenamiento)
    feature_names = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "hour_of_week",
        "is_holiday",
        "is_holiday_eve",
        "lag_1h",
        "lag_2h",
        "lag_3h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_3h",
        "rolling_mean_12h",
        "rolling_mean_24h",
        "rolling_mean_168h",
    ]

    X = np.array([[features[name] for name in feature_names]])
    pred = model.predict(X)[0]

    return max(0.0, pred)  # No predicciones negativas


def predict_next_24h(
    zone_id: int,
    start_date: pd.Timestamp,
    history_df: pd.DataFrame,
    model,
) -> list[dict]:
    """Predecir demanda de las próximas 24 horas (hora a hora).

    Returns:
        Lista de dicts: [{'hour': 0, 'prediction': 5.2}, {'hour': 1, 'prediction': 3.1}, ...]
    """
    predictions = []

    for h in range(24):
        dt = start_date + pd.Timedelta(hours=h)
        pred = predict_demand(zone_id, dt, history_df, model)
        predictions.append(
            {
                "hour": h,
                "datetime": dt,
                "prediction": pred,
            }
        )

    return predictions
