import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="NYC Taxi Demand", page_icon="🚕", layout="wide")

st.title("🚕 NYC Taxi Demand Predictor")
st.markdown("Modelo: Yellow + FHVHV (2023-2025) | Linear Regression | MAE: 0.264 viajes/hora")

try:
    model = joblib.load('models/linear_model_combined.joblib')
    history = pd.read_parquet('data/processed/history_recent_combined.parquet')
    st.success("✓ Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

# Barra lateral
st.sidebar.header("Parametros")
zone_id = st.sidebar.slider("Zona (1-265):", 1, 265, 237)
target_hour = st.sidebar.slider("Hora (0-23):", 0, 23, 14)
view_mode = st.sidebar.radio("Modo:", ["Individual", "24 horas"])

# Calcular features manualmente
from src.app.predict import get_features_for_datetime, predict_demand, predict_next_24h

target_dt = pd.Timestamp(year=2025, month=1, day=15, hour=target_hour)
history['pickup_datetime'] = pd.to_datetime(history['pickup_datetime'])

# VISTA INDIVIDUAL
if view_mode == "Individual":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Zona", zone_id)
    with col2:
        st.metric("Hora", f"{target_hour:02d}:00")
    with col3:
        pred = predict_demand(zone_id, target_dt, history, model)
        st.metric("Prediccion", f"{pred:.1f}", delta="viajes/h", delta_color="off")
    
    st.info(f"Intervalo (±20%): {pred*0.8:.1f} - {pred*1.2:.1f} viajes/hora")

# VISTA 24H
else:
    preds = predict_next_24h(zone_id, target_dt, history, model)
    df = pd.DataFrame(preds)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in df['hour']],
        y=df['prediction'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(height=400, title=f"Zona {zone_id} - Proximas 24h", template='plotly_light')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max", f"{df['prediction'].max():.1f}")
    with col2:
        st.metric("Min", f"{df['prediction'].min():.1f}")
    with col3:
        st.metric("Prom", f"{df['prediction'].mean():.1f}")
    with col4:
        st.metric("Total", f"{df['prediction'].sum():.0f}")

st.markdown("---")
st.caption("Dataset: Yellow + FHVHV (2023-2025) | Modelo: Linear Regression")
