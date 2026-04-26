import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from src.app.predict import get_features_for_datetime, predict_demand, predict_next_24h

# ============================================================================
# CONFIG Y ESTILOS
# ============================================================================

st.set_page_config(
    page_title="NYC Taxi Demand Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f0f0f; color: #ffffff; }
    .stMetricValue { font-size: 2.2rem; font-weight: bold; color: #0066cc; }
    .stMetricLabel { color: #b0b0b0; font-size: 0.9rem; font-weight: 500; }
    h1, h2, h3 { color: #ffffff; font-weight: 600; }
    .info-card {
        background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
        padding: 20px;
        border-radius: 8px;
        color: white;
        border-left: 4px solid #0080ff;
    }
    .stat-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #404040;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGAR DATOS
# ============================================================================

@st.cache_resource
def load_data():
    try:
        model = joblib.load('models/linear_model_combined.joblib')
        history = pd.read_parquet('data/processed/history_recent_combined.parquet')
        zones = pd.read_csv('data/external/taxi_zone_lookup.csv')
        history['pickup_datetime'] = pd.to_datetime(history['pickup_datetime'])
        return model, history, zones, None
    except Exception as e:
        return None, None, None, str(e)

model, history_df, zones_df, error = load_data()

if model is None:
    st.error(f"Error: {error}")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([0.75, 0.25])
with col1:
    st.title("🚕 NYC Taxi Demand Predictor")
    st.markdown("Prediccion inteligente de demanda de taxis en Nueva York")
with col2:
    st.markdown("###")
    st.info("Modelo entrenado\nDatos: 2023-2025\nMAE: 0.264")

st.markdown("---")

# ============================================================================
# PARÁMETROS EN SIDEBAR
# ============================================================================

st.sidebar.title("Parametros")

# Crear diccionario de zonas
zones_dict = {}
for _, row in zones_df.iterrows():
    zone_id = int(row['LocationID'])
    zone_name = row['Zone']
    borough = row['Borough']
    zones_dict[f"{zone_id:3d} - {zone_name} ({borough})"] = zone_id

st.sidebar.subheader("Zona")
zone_label = st.sidebar.selectbox(
    "Selecciona una zona",
    options=sorted(zones_dict.keys()),
    index=136
)
zone_id = zones_dict[zone_label]

st.sidebar.subheader("Hora")
target_hour = st.sidebar.slider(
    "Selecciona la hora del dia",
    min_value=0,
    max_value=23,
    value=14,
    format="%02d:00"
)

st.sidebar.subheader("Modo")
view_mode = st.sidebar.radio(
    "Tipo de visualizacion",
    options=["Prediccion Individual", "Proximas 24 Horas"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Info del Modelo**
- Algoritmo: Linear Regression
- Dataset: Yellow + FHVHV
- Periodo: 2023-2025
- Exactitud: MAE 0.264
""")

# ============================================================================
# CÁLCULOS
# ============================================================================

target_dt = pd.Timestamp(year=2025, month=1, day=15, hour=target_hour)

# ============================================================================
# VISTA 1: INDIVIDUAL
# ============================================================================

if view_mode == "Prediccion Individual":

    zone_info = zones_df[zones_df['LocationID'] == zone_id].iloc[0]

    st.markdown(f"### Prediccion - {zone_label}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Zona", zone_id)
    with col2:
        st.metric("Barrio", zone_info['Borough'])
    with col3:
        st.metric("Hora", f"{target_hour:02d}:00")
    with col4:
        pred = predict_demand(zone_id, target_dt, history_df, model)
        st.metric("Demanda", f"{pred:.1f}", delta="viajes/h", delta_color="off")

    st.markdown("---")
    st.markdown("### Analisis de Precision")

    col1, col2, col3 = st.columns(3)

    with col1:
        lower = pred * 0.8
        upper = pred * 1.2
        st.markdown(f"""
        <div class='info-card'>
            <strong>Intervalo de Confianza (±20%)</strong><br><br>
            <span style='font-size: 1.8em;'>{lower:.1f} - {upper:.1f}</span><br>
            <span>viajes/hora</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <strong>Precision</strong><br><br>
            <span style='color: #0066cc; font-size: 1.6em;'>MAE: 0.264</span><br>
            <span style='color: #b0b0b0;'>viajes/hora</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <strong>Ajuste (R²)</strong><br><br>
            <span style='color: #0066cc; font-size: 1.6em;'>0.9999</span><br>
            <span style='color: #b0b0b0;'>Excelente</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Caracteristicas Temporales")
        features = get_features_for_datetime(target_dt, zone_id, history_df)
        days = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
        st.markdown(f"""
        - **Hora:** {features['hour']:02d}:00
        - **Dia:** {days[features['day_of_week']]}
        - **Fin de semana:** {'Si' if features['is_weekend'] else 'No'}
        - **Festivo:** {'Si' if features['is_holiday'] else 'No'}
        """)

    with col2:
        st.markdown("### Lags y Promedios")
        st.markdown(f"""
        - **Lag 1h:** {features['lag_1h']:.1f} viajes
        - **Lag 24h:** {features['lag_24h']:.1f} viajes
        - **Lag 7 dias:** {features['lag_168h']:.1f} viajes
        - **Prom movil 24h:** {features['rolling_mean_24h']:.1f} viajes
        """)

# ============================================================================
# VISTA 2: 24 HORAS
# ============================================================================

else:
    st.markdown(f"### Proximas 24 horas - {zone_label}")

    preds = predict_next_24h(zone_id, target_dt, history_df, model)
    df = pd.DataFrame(preds)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in df['hour']],
        y=df['prediction'],
        mode='lines+markers',
        name='Prediccion',
        line=dict(color='#0066cc', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.2)'
    ))

    fig.update_layout(
        title="Demanda por hora",
        xaxis_title="Hora del dia",
        yaxis_title="Viajes/hora",
        hovermode='x unified',
        height=450,
        template='plotly_dark',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0f0f0f',
        font=dict(color='#ffffff')
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    vals = df['prediction'].values

    with col1:
        st.metric("Maximo", f"{vals.max():.1f}", delta="viajes/h", delta_color="off")
    with col2:
        st.metric("Minimo", f"{vals.min():.1f}", delta="viajes/h", delta_color="off")
    with col3:
        st.metric("Promedio", f"{vals.mean():.1f}", delta="viajes/h", delta_color="off")
    with col4:
        st.metric("Total (24h)", f"{vals.sum():.0f}", delta="viajes", delta_color="off")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #707070; padding: 20px;'>
    <p><strong>NYC Taxi Demand Predictor</strong> | Yellow + FHVHV (2023-2025)</p>
    <p>Linear Regression | MAE: 0.264 | R²: 0.9999</p>
</div>
""", unsafe_allow_html=True)
