"""NYC Taxi Demand Predictor - Streamlit Web App."""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.predict import predict_demand, predict_next_24h


@st.cache_resource
def load_data_and_model():
    """Cargar modelo, histórico, datos completos y lookup de zonas."""
    model_path = Path("models/linear_model.joblib")
    history_path = Path("data/processed/history_recent.parquet")
    zones_path = Path("data/external/taxi_zone_lookup.csv")
    full_data_path = Path("data/processed/yellow_taxi_features.parquet")

    model = joblib.load(model_path)
    history_df = pd.read_parquet(history_path)
    zones_df = pd.read_csv(zones_path)
    full_df = pd.read_parquet(full_data_path)

    # Convertir timestamps
    if "pickup_datetime" in history_df.columns:
        history_df["pickup_datetime"] = pd.to_datetime(history_df["pickup_datetime"])
    if "pickup_datetime" in full_df.columns:
        full_df["pickup_datetime"] = pd.to_datetime(full_df["pickup_datetime"])

    return model, history_df, zones_df, full_df


def create_zone_options(zones_df):
    """Crear opciones para el dropdown de zonas con formato amigable."""
    options = {}
    for _, row in zones_df.iterrows():
        zone_id = int(row["LocationID"])
        zone_name = row["Zone"]
        display = f"{zone_id:3d} — {zone_name}"
        options[display] = zone_id
    return options


def plot_24h_forecast(predictions):
    """Crear gráfico de barras con predicción de 24 horas."""
    df = pd.DataFrame(predictions)

    # Crear color: amarillo para la hora actual, gris para el resto
    colors = ["#F6C90E" if h == df.iloc[0]["hour"] else "#4B5563" for h in df["hour"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["hour"],
                y=df["prediction"],
                marker=dict(color=colors),
                text=df["prediction"].round(1),
                textposition="auto",
                hovertemplate="<b>Hora %{x}:00</b><br>Viajes: %{y:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Demanda predicha - Próximas 24 horas",
        xaxis_title="Hora del día",
        yaxis_title="Viajes/hora",
        template="plotly_dark",
        height=400,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        paper_bgcolor="#16213E",
        plot_bgcolor="#16213E",
        font=dict(color="#EAEAEA"),
    )

    return fig


def plot_historical_comparison(zone_id, history_df, model, target_date, num_days=7):
    """Comparar predicción vs datos históricos de los últimos días."""
    # Obtener datos históricos de la zona
    zone_history = history_df[history_df["PULocationID"] == zone_id].copy()

    if len(zone_history) == 0:
        st.warning("No hay datos históricos para esta zona.")
        return None

    zone_history["pickup_datetime"] = pd.to_datetime(zone_history["pickup_datetime"])
    zone_history = zone_history.sort_values("pickup_datetime")

    # Últimos 7 días desde el objetivo hacia atrás
    start_date = target_date - timedelta(days=num_days)
    recent = zone_history[zone_history["pickup_datetime"] >= start_date].copy()

    if len(recent) == 0:
        st.warning("No hay datos históricos para el período seleccionado.")
        return None

    # Agregar por día
    recent["date"] = recent["pickup_datetime"].dt.date
    daily = recent.groupby("date")["trip_count"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Calcular predicción para el día objetivo
    pred_next_24h = predict_next_24h(zone_id, target_date, history_df, model)
    pred_sum = sum([p["prediction"] for p in pred_next_24h])

    # Crear figura
    fig = go.Figure()

    # Línea de histórico real
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["trip_count"],
            mode="lines+markers",
            name="Real (histórico)",
            line=dict(color="#4B9FEE", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%d %b}</b><br>Viajes: %{y:.0f}<extra></extra>",
        )
    )

    # Añadir punto de predicción para la fecha objetivo
    fig.add_trace(
        go.Scatter(
            x=[target_date],
            y=[pred_sum],
            mode="markers+text",
            name="Predicción",
            marker=dict(size=12, color="#F6C90E", symbol="star"),
            text=["Predicción"],
            textposition="top center",
            hovertemplate="<b>%{x|%d %b} (Predicción)</b><br>Viajes: %{y:.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Histórico vs Predicción (agregado diario)",
        xaxis_title="Fecha",
        yaxis_title="Viajes/día",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        paper_bgcolor="#16213E",
        plot_bgcolor="#16213E",
        font=dict(color="#EAEAEA"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def plot_pred_vs_real_24h(zone_id, selected_date, history_df, full_df, model):
    """Gráfico de predicción vs real para un día completo."""
    # Obtener predicciones
    predictions_24h = predict_next_24h(zone_id, selected_date, history_df, model)
    pred_df = pd.DataFrame(predictions_24h)

    # Obtener datos reales
    real_data = full_df[
        (full_df["PULocationID"] == zone_id)
        & (full_df["pickup_datetime"].dt.date == selected_date.date())
    ].copy()

    if len(real_data) == 0:
        return None

    real_data = real_data.sort_values("pickup_datetime")
    real_by_hour = (
        real_data.groupby(real_data["pickup_datetime"].dt.hour)["trip_count"].first().reset_index()
    )
    real_by_hour.columns = ["hour", "trip_count"]

    # Merge con predicciones
    comparison = pred_df.merge(real_by_hour, on="hour", how="left")
    comparison["trip_count"] = comparison["trip_count"].fillna(0)

    # Crear figura
    fig = go.Figure()

    # Barras de predicción
    fig.add_trace(
        go.Bar(
            x=comparison["hour"],
            y=comparison["prediction"],
            name="Predicción",
            marker=dict(color="#F6C90E"),
            opacity=0.7,
            hovertemplate="<b>Hora %{x}:00</b><br>Predicción: %{y:.2f}<extra></extra>",
        )
    )

    # Línea de real
    fig.add_trace(
        go.Scatter(
            x=comparison["hour"],
            y=comparison["trip_count"],
            name="Real",
            mode="lines+markers",
            line=dict(color="#4B9FEE", width=3),
            marker=dict(size=8),
            hovertemplate="<b>Hora %{x}:00</b><br>Real: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Predicción vs Real (día completo)",
        xaxis_title="Hora del día",
        yaxis_title="Viajes/hora",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        barmode="overlay",
        paper_bgcolor="#16213E",
        plot_bgcolor="#16213E",
        font=dict(color="#EAEAEA"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )

    return fig


def main():
    st.set_page_config(
        page_title="NYC Taxi Demand Predictor",
        page_icon="🗽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS personalizado para mejorar estética
    st.markdown(
        """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #16213E 0%, #0F3460 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #F6C90E;
    }
    .header-title {
        color: #F6C90E;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .header-subtitle {
        color: #EAEAEA;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .period-badge {
        display: inline-block;
        background: #16213E;
        border: 1px solid #F6C90E;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 4px 4px 0;
        font-size: 0.85em;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        '<div class="header-title">🗽 NYC Taxi Demand Predictor</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="header-subtitle">Predicción de demanda de taxis por zona y hora</div>',
        unsafe_allow_html=True,
    )

    # Cargar datos
    with st.spinner("Cargando modelo y datos..."):
        model, history_df, zones_df, full_df = load_data_and_model()
        zone_options = create_zone_options(zones_df)

    # Info del modelo con rangos de entrenamiento y test
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Modelo", "Linear Regression")
    with col2:
        st.metric("MAE Test", "0.1447")
    with col3:
        st.metric("R² Score", "0.9999")
    with col4:
        st.metric("Entrenamiento", "2024-01-01\na 2025-02-28")
    with col5:
        st.metric("Test", "2025-03-01\na 2025-03-31")

    st.divider()

    # Mostrar claramente los períodos de datos
    st.info(
        "📊 **Períodos de datos disponibles:**\n\n"
        "🟢 **ENTRENAMIENTO:** 1 de enero 2024 — 28 de febrero 2025 (Modelo entrenado con estos datos)\n\n"
        "🔵 **TEST:** 1 de marzo 2025 — 31 de marzo 2025 (Validación del modelo; puedes ver predicción vs real)\n\n"
        "⚪ **FUTURO:** Desde 1 de abril 2025 en adelante (Solo predicción)"
    )

    st.divider()

    # Sidebar - Inputs
    with st.sidebar:
        st.subheader("⚙️ Parámetros de predicción")

        # Selector de zona
        selected_zone_display = st.selectbox(
            "Selecciona una zona NYC:",
            options=list(zone_options.keys()),
            index=0,
            help="Selecciona la zona de pickup del taxi",
        )
        zone_id = zone_options[selected_zone_display]

        # Date picker - ahora permite cualquier fecha
        min_date = pd.Timestamp("2024-01-01")
        max_date = datetime.now() + timedelta(days=365)
        selected_date = st.date_input(
            "Fecha:",
            value=datetime.now(),
            min_value=min_date.date(),
            max_value=max_date.date(),
            help="Selecciona una fecha (pasada o futura)",
        )
        selected_datetime = pd.Timestamp(selected_date)

        # Hour slider
        selected_hour = st.slider(
            "Hora del día:",
            min_value=0,
            max_value=23,
            value=12,
            step=1,
            format="%d:00",
            help="Selecciona la hora del día (0-23)",
        )

        st.divider()

        # Mostrar tipo de fecha seleccionada
        if selected_datetime < pd.Timestamp("2024-01-01"):
            st.error("❌ Fecha antes del inicio de datos")
        elif selected_datetime < pd.Timestamp("2025-03-01"):
            st.warning("⚠️ Fecha en período de ENTRENAMIENTO\n(sin datos de test disponibles)")
        elif selected_datetime < pd.Timestamp("2025-04-01"):
            st.success("✅ Fecha en período de TEST\n(puedes ver predicción vs real)")
        else:
            st.info("🔮 Fecha FUTURA\n(solo predicción disponible)")

        st.divider()

        # Botón de predicción
        st.info('👆 Selecciona los parámetros y presiona "Predecir"')

    # Main content - Results
    if st.sidebar.button("Predecir", type="primary", use_container_width=True):
        # Crear timestamp de la predicción
        target_datetime = selected_datetime + pd.Timedelta(hours=selected_hour)

        # Información de la zona
        zone_info = zones_df[zones_df["LocationID"] == zone_id].iloc[0]
        zone_name = zone_info["Zone"]

        # Mostrar zona
        st.subheader(f"Zona {zone_id} — {zone_name}")

        # Determinar si hay datos reales disponibles
        has_real_data = not full_df[
            (full_df["PULocationID"] == zone_id) & (full_df["pickup_datetime"] == target_datetime)
        ].empty

        # Hacer predicción
        prediction = predict_demand(zone_id, target_datetime, history_df, model)

        # Si es una fecha con datos reales, mostrar comparativa
        if has_real_data:
            real_value = full_df[
                (full_df["PULocationID"] == zone_id)
                & (full_df["pickup_datetime"] == target_datetime)
            ]["trip_count"].iloc[0]

            error = abs(prediction - real_value)
            mape = 100 * error / (real_value + 1)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Predicción",
                    f"{prediction:.2f}",
                    delta=None,
                    help="Valor predicho por el modelo",
                )

            with col2:
                st.metric("Real", f"{real_value:.2f}", delta=None, help="Valor real observado")

            with col3:
                st.metric(
                    "Error Absoluto",
                    f"{error:.2f}",
                    delta=None,
                    help="Diferencia entre predicción y real",
                )

            with col4:
                st.metric(
                    "MAPE (%)", f"{mape:.1f}%", delta=None, help="Error Porcentual Absoluto Medio"
                )

            # Badge de período
            st.success("✅ **DATOS REALES DISPONIBLES** — Período de test (marzo 2025)")

        else:
            col_metric, col_info = st.columns([2, 1])

            with col_metric:
                st.metric(
                    "Viajes esperados/hora",
                    f"{prediction:.2f}",
                    delta=None,
                    help="Predicción de demanda de taxis",
                )

            with col_info:
                st.write(f'📅 {target_datetime.strftime("%d de %b de %Y")}')
                st.write(f'🕒 {target_datetime.strftime("%H:%M")}')

        st.divider()

        # Gráfico de 24 horas
        st.subheader("📊 Demanda por hora — Día completo")

        # Si tiene datos reales, mostrar comparativa
        if selected_datetime >= pd.Timestamp("2025-03-01") and selected_datetime < pd.Timestamp(
            "2025-04-01"
        ):
            fig_24h = plot_pred_vs_real_24h(zone_id, selected_datetime, history_df, full_df, model)
            if fig_24h:
                st.plotly_chart(fig_24h, use_container_width=True)
            else:
                st.warning("No hay datos reales disponibles para este día.")
        else:
            # Solo mostrar predicción
            predictions_24h = predict_next_24h(zone_id, selected_datetime, history_df, model)
            fig_24h = plot_24h_forecast(predictions_24h)
            st.plotly_chart(fig_24h, use_container_width=True)

        # Comparación histórica
        st.subheader("📈 Comparación con histórico")
        fig_historical = plot_historical_comparison(
            zone_id, history_df, model, selected_datetime, num_days=7
        )
        if fig_historical:
            st.plotly_chart(fig_historical, use_container_width=True)

        # Estadísticas de la zona
        st.subheader("📋 Estadísticas de la zona")
        zone_data = history_df[history_df["PULocationID"] == zone_id]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Promedio viajes/hora", f'{zone_data["trip_count"].mean():.1f}')
        with col2:
            st.metric("Máximo viajes/hora", f'{zone_data["trip_count"].max():.0f}')
        with col3:
            st.metric("Mínimo viajes/hora", f'{zone_data["trip_count"].min():.0f}')
        with col4:
            st.metric("Registros históricos", f"{len(zone_data)}")

    else:
        # Estado inicial - mostrar instrucciones
        st.info(
            "👈 Usa los controles del sidebar para seleccionar una zona, fecha y hora. "
            'Luego presiona "Predecir" para ver la demanda esperada de taxis.'
        )

        # Mostrar información general
        st.subheader("ℹ️ Acerca de este predictor")
        st.write("""
        Este predictor utiliza un modelo de **Linear Regression** entrenado con datos
        de taxis amarillos de NYC.

        **Características del modelo:**
        - 261 zonas de pickup (LocationID)
        - Features temporales: hora, día de la semana, mes, festivos
        - Lags (retardos): 1h, 2h, 3h, 24h, 7 días
        - Medias móviles: 3h, 12h, 24h, 7 días

        **Datos disponibles:**
        - 🟢 **Entrenamiento:** 1 ene 2024 — 28 feb 2025 (1.8M registros)
        - 🔵 **Validación:** 1 mar — 31 mar 2025 (188k registros)

        **Precisión en test:**
        - MAE (Error Absoluto Medio): 0.1447 viajes/hora
        - RMSE: 0.4596
        - R² Score: 0.9999 (99.99% de varianza explicada)
        """)

        st.subheader("💡 Cómo usar")
        st.write("""
        1. **Selecciona una zona** de pickup (ej: Midtown Center, JFK Airport)
        2. **Elige una fecha:**
           - ✅ **Marzo 2025** (test): Verás predicción comparada con datos reales
           - 🔮 **Fechas futuras**: Solo predicción
        3. **Selecciona la hora** (0-23)
        4. **Presiona "Predecir"** para ver el resultado

        Si eliges una fecha en el período de test, podrás validar qué tan bueno
        es el modelo comparando predicción vs valores reales.
        """)

        st.subheader("🗺️ Zonas disponibles")
        st.write(f"Total de zonas: {len(zones_df)}")

        # Mostrar tabla de zonas (primeras 10)
        with st.expander("Ver primeras 10 zonas"):
            st.dataframe(zones_df.head(10), hide_index=True)


if __name__ == "__main__":
    main()
