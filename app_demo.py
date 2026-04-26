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

        # Cargar datos completos para comparación real vs predicción
        features = pd.read_parquet('data/processed/yellow_taxi_features_combined.parquet')

        history['pickup_datetime'] = pd.to_datetime(history['pickup_datetime'])
        features['pickup_datetime'] = pd.to_datetime(features['pickup_datetime'])

        return model, history, zones, features, None
    except Exception as e:
        return None, None, None, None, str(e)

model, history_df, zones_df, features_df, error = load_data()

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

# FECHA
st.sidebar.subheader("Fecha")
min_date = features_df['pickup_datetime'].min().date()
max_date = features_df['pickup_datetime'].max().date()
future_date = max_date + pd.Timedelta(days=365)

target_date = st.sidebar.date_input(
    "Selecciona la fecha",
    value=max_date,
    min_value=min_date,
    max_value=future_date,
    help=f"Rango: {min_date} a {future_date.date()}"
)

# ZONA
st.sidebar.subheader("Zona")
zone_label = st.sidebar.selectbox(
    "Selecciona una zona",
    options=sorted(zones_dict.keys()),
    index=136
)
zone_id = zones_dict[zone_label]

# HORA
st.sidebar.subheader("Hora")
target_hour = st.sidebar.slider(
    "Selecciona la hora del dia",
    min_value=0,
    max_value=23,
    value=14,
    format="%02d:00"
)

# MODO
st.sidebar.subheader("Modo")
view_mode = st.sidebar.radio(
    "Tipo de visualizacion",
    options=["Prediccion Individual", "Proximas 24 Horas"],
    index=0
)

# Determinar si es fecha histórica o futura
target_dt = pd.Timestamp(year=target_date.year, month=target_date.month, day=target_date.day, hour=target_hour)
is_historical = target_dt <= pd.Timestamp(max_date)

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

# Obtener predicción
pred = predict_demand(zone_id, target_dt, history_df, model)

# Obtener dato real si existe
real_value = None
if is_historical:
    real_data = features_df[
        (features_df['pickup_datetime'] == target_dt) &
        (features_df['PULocationID'] == zone_id)
    ]
    if not real_data.empty:
        real_value = real_data.iloc[0]['trip_count']

# ============================================================================
# VISTA 1: INDIVIDUAL
# ============================================================================

if view_mode == "Prediccion Individual":

    zone_info = zones_df[zones_df['LocationID'] == zone_id].iloc[0]

    status_label = "Histórico" if is_historical else "Futuro"
    st.markdown(f"### {status_label} - {zone_label}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Zona", zone_id)
    with col2:
        st.metric("Barrio", zone_info['Borough'])
    with col3:
        st.metric("Fecha/Hora", f"{target_date.strftime('%d/%m %H:%M')}")
    with col4:
        if is_historical and real_value is not None:
            st.metric("Real", f"{real_value:.1f}", delta="viajes/h", delta_color="off")
        else:
            st.metric("Estado", "Futuro", help="Fecha sin datos reales")

    st.markdown("---")

    # Comparación Real vs Predicción (si es histórico)
    if is_historical and real_value is not None:
        st.markdown("### Comparacion Real vs Prediccion")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='stat-card'>
                <strong>Dato Real</strong><br><br>
                <span style='color: #00cc00; font-size: 1.8em;'>{real_value:.1f}</span><br>
                <span style='color: #b0b0b0;'>viajes/hora</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='stat-card'>
                <strong>Prediccion</strong><br><br>
                <span style='color: #0066cc; font-size: 1.8em;'>{pred:.1f}</span><br>
                <span style='color: #b0b0b0;'>viajes/hora</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            error_abs = abs(real_value - pred)
            error_pct = (error_abs / (real_value + 0.001)) * 100
            color = "#ff6600" if error_pct > 10 else "#ffaa00" if error_pct > 5 else "#00cc00"
            st.markdown(f"""
            <div class='stat-card'>
                <strong>Error</strong><br><br>
                <span style='color: {color}; font-size: 1.6em;'>{error_pct:.1f}%</span><br>
                <span style='color: #b0b0b0;'>({error_abs:.1f} viajes)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    st.markdown("### Analisis de Precision")

    col1, col2, col3 = st.columns(3)

    with col1:
        lower = pred * 0.8
        upper = pred * 1.2
        st.markdown(f"""
        <div class='info-card'>
            <strong>Prediccion</strong><br><br>
            <span style='font-size: 1.8em;'>{pred:.1f}</span><br>
            <span>viajes/hora</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <strong>Intervalo ±20%</strong><br><br>
            <span style='color: #0066cc; font-size: 1.4em;'>{lower:.1f} - {upper:.1f}</span><br>
            <span style='color: #b0b0b0;'>viajes/hora</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <strong>Precision (MAE)</strong><br><br>
            <span style='color: #0066cc; font-size: 1.6em;'>0.264</span><br>
            <span style='color: #b0b0b0;'>viajes/hora</span>
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
