import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from dotenv import load_dotenv
import hopsworks
import joblib
import plotly.graph_objects as go
import streamlit.components.v1 as components

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_pipeline import add_derived_features

load_dotenv()

st.set_page_config(page_title="Islamabad AQI Forecast", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: #f6f3ef;
    }

    .block-container {
        position: relative;
        z-index: 2;
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 920px;
    }

    .app-title {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: #2d2a26;
        margin-bottom: 0;
    }

    .app-subtitle {
        font-family: 'Nunito', sans-serif;
        color: #9b9691;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 1.8rem;
    }

    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4cd68a;
        margin-right: 6px;
        animation: livePulse 2s ease-in-out infinite;
    }
    @keyframes livePulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(76,214,138,0.5); }
        50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(76,214,138,0); }
    }

    .section-label {
        font-family: 'Baloo 2', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d2a26;
        margin-bottom: 14px;
        margin-top: 12px;
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    div[class*="st-key-gauge-card"] {
        border-radius: 28px !important;
        padding: 16px 16px 4px 16px !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08) !important;
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-delay: 0.15s;
    }

    div[class*="st-key-stats-card"] {
        background: #ffffff !important;
        border-radius: 28px !important;
        padding: 18px 20px !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08) !important;
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-delay: 0.25s;
    }

    div[class*="st-key-chart-card"] {
        background: #ffffff !important;
        border-radius: 28px !important;
        padding: 20px !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08) !important;
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-delay: 0.4s;
    }

    .icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-size: 1.1rem;
        margin-right: 12px;
        flex-shrink: 0;
        animation: floatBounce 3s ease-in-out infinite;
    }
    @keyframes floatBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
    }

    .forecast-card {
        border-radius: 26px;
        padding: 24px 18px;
        text-align: center;
        transition: transform 0.25s ease;
        height: 100%;
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-delay: 0.3s;
    }
    .forecast-card:hover {
        transform: translateY(-5px) scale(1.02);
    }

    .forecast-day {
        font-family: 'Nunito', sans-serif;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 6px;
    }

    .forecast-number {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 700;
        line-height: 1;
        animation: numberPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        animation-delay: 0.5s;
    }

    @keyframes numberPop {
        0% { transform: scale(0.7); opacity: 0; }
        60% { transform: scale(1.08); opacity: 1; }
        100% { transform: scale(1); }
    }

    .forecast-cat {
        font-family: 'Nunito', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 8px;
        opacity: 0.9;
    }

    .alert-banner {
        border-radius: 26px;
        padding: 22px 28px;
        margin-top: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        background: #ff6b6b;
        box-shadow: 0 10px 24px rgba(255,107,107,0.3);
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes sparkleFloat {
        0% { transform: translateY(0) rotate(0deg); opacity: 0.8; }
        100% { transform: translateY(-60px) rotate(180deg); opacity: 0; }
    }
    .sparkle {
        position: absolute;
        font-size: 1.2rem;
        animation: sparkleFloat 2.5s ease-out infinite;
        pointer-events: none;
    }
    </style>
""", unsafe_allow_html=True)


def particle_background(color):
    components.html(f"""
    <div id="particle-container" style="position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:1; pointer-events:none;">
        <canvas id="particles"></canvas>
    </div>
    <script>
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleColor = "{color}";

    for (let i = 0; i < 45; i++) {{
        particles.push({{
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 3 + 1,
            speedX: (Math.random() - 0.5) * 0.4,
            speedY: (Math.random() - 0.5) * 0.4,
            opacity: Math.random() * 0.4 + 0.15
        }});
    }}

    function animate() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {{
            p.x += p.speedX;
            p.y += p.speedY;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = particleColor + Math.floor(p.opacity * 255).toString(16).padStart(2, '0');
            ctx.fill();
        }});
        requestAnimationFrame(animate);
    }}
    animate();

    window.addEventListener('resize', () => {{
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }});
    </script>
    """, height=0, width=0)


def get_aqi_category(pm25):
    if pm25 <= 50:
        return "Good", "#2d7a4f", "#a8e6c1"
    elif pm25 <= 100:
        return "Moderate", "#8a6d1a", "#ffe17d"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive Groups", "#a34d0f", "#ffb26b"
    elif pm25 <= 200:
        return "Unhealthy", "#ffffff", "#ff6b6b"
    elif pm25 <= 300:
        return "Very Unhealthy", "#ffffff", "#9d6bd6"
    else:
        return "Hazardous", "#ffffff", "#7a2e4a"


def make_gauge(value, color, max_val=300):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 54, 'family': 'Baloo 2', 'color': color}, 'suffix': ""},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)", 'showticklabels': False},
            'bar': {'color': color, 'thickness': 0.32},
            'bgcolor': "rgba(255,255,255,0.35)",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': color}
    )
    return fig


FEATURE_COLUMNS = [
    "hour", "day", "month_sin", "month_cos", "pm25", "temp", "humidity", "pressure", "wind_speed",
    "pm25_lag_1h", "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_72h", "pm25_change_rate"
]


@st.cache_resource
def connect():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    return project


@st.cache_resource
def load_model(_project, target_name):
    mr = _project.get_model_registry()
    model_meta = mr.get_model(f"aqi_ridge_{target_name}", version=2)
    model_dir = model_meta.download()
    model_path = os.path.join(model_dir, "model.pkl")
    return joblib.load(model_path)


def load_feature_data(fs, retries=3, delay=5):
    last_error = None
    for attempt in range(retries):
        try:
            fg = fs.get_feature_group(name="aqi_features_v2", version=1)
            df = fg.read()
            return df
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                st.warning(f"Retrying data load... ({attempt + 1}/{retries})")
                time.sleep(delay)
    st.error("Unable to load data from Hopsworks after several attempts. Please refresh the page in a moment.")
    st.stop()


project = connect()
fs = project.get_feature_store()

df = load_feature_data(fs)
df = df.sort_values("timestamp").reset_index(drop=True)
df = add_derived_features(df)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

latest = df.tail(1)
current_pm25 = latest["pm25"].values[0]
category, text_color, bg_color = get_aqi_category(current_pm25)

particle_background(bg_color)

model_24h = load_model(project, "target_24h")
model_48h = load_model(project, "target_48h")
model_72h = load_model(project, "target_72h")

X_latest = latest[FEATURE_COLUMNS]
pred_24h = model_24h.predict(X_latest)[0]
pred_48h = model_48h.predict(X_latest)[0]
pred_72h = model_72h.predict(X_latest)[0]

st.markdown('<p class="app-title">🌤️ Islamabad AQI Forecast</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="app-subtitle"><span class="live-dot"></span>LIVE · UPDATED '
    f'{pd.to_datetime(latest["timestamp"].values[0]).strftime("%b %d, %H:%M").upper()}</p>',
    unsafe_allow_html=True
)

if category == "Good":
    st.markdown(
        """
        <div style="position:relative; height:0;">
            <span class="sparkle" style="left:10%; top:-20px; animation-delay:0s;">✨</span>
            <span class="sparkle" style="left:30%; top:-10px; animation-delay:0.5s;">⭐</span>
            <span class="sparkle" style="left:60%; top:-25px; animation-delay:1s;">✨</span>
            <span class="sparkle" style="left:80%; top:-15px; animation-delay:1.5s;">⭐</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<p class="section-label">✨ Current Conditions</p>', unsafe_allow_html=True)
st.markdown(f'<style>div[class*="st-key-gauge-card"] {{ background:{bg_color} !important; }}</style>', unsafe_allow_html=True)

col_gauge, col_stats = st.columns([1.1, 1])

with col_gauge:
    with st.container(key="gauge-card"):
        st.plotly_chart(make_gauge(current_pm25, text_color), use_container_width=True, key="aqi_gauge")
        st.markdown(
            f'<p style="text-align:center; margin-top:-16px; color:{text_color}; font-family:Baloo 2; font-weight:700; font-size:1.2rem;">{category}</p>'
            f'<p style="text-align:center; font-size:0.8rem; color:{text_color}; opacity:0.75; margin-top:2px; margin-bottom:18px; font-weight:600;">PM2.5 · μg/m³</p>',
            unsafe_allow_html=True
        )

with col_stats:
    temp = latest["temp"].values[0]
    humidity = latest["humidity"].values[0]
    wind = latest["wind_speed"].values[0]
    pressure = latest["pressure"].values[0]

    stat_colors = ["#ffd6d6", "#d6ecff", "#d6ffe0", "#f0d6ff"]
    stats = [
        ("🌡️", "Temperature", f"{temp:.1f}°C"),
        ("💧", "Humidity", f"{humidity:.0f}%"),
        ("🍃", "Wind Speed", f"{wind:.1f} m/s"),
        ("🎈", "Pressure", f"{pressure:.0f} hPa"),
    ]

    with st.container(key="stats-card"):
        for (icon, label, val), badge_color in zip(stats, stat_colors):
            st.markdown(
                f'<div class="stat-row">'
                f'<div style="display:flex; align-items:center;">'
                f'<span class="icon-badge" style="background:{badge_color};">{icon}</span>'
                f'<span style="font-family:Nunito; color:#5c5652; font-size:0.9rem; font-weight:700;">{label}</span>'
                f'</div>'
                f'<span style="font-family:Baloo 2; color:#2d2a26; font-weight:700; font-size:1.05rem;">{val}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

st.markdown('<p class="section-label" style="margin-top:32px;">🔮 3-Day Forecast</p>', unsafe_allow_html=True)

labels = ["Tomorrow", "In 2 days", "In 3 days"]
values = [pred_24h, pred_48h, pred_72h]
worst_idx = max(range(3), key=lambda i: values[i])

if worst_idx == 0:
    widths = [1.4, 1, 1]
elif worst_idx == 1:
    widths = [1, 1.4, 1]
else:
    widths = [1, 1, 1.4]

bento_cols = st.columns(widths)

for i, (col, label, value) in enumerate(zip(bento_cols, labels, values)):
    cat, txt_c, bg_c = get_aqi_category(value)
    size = "2.6rem" if i == worst_idx else "1.9rem"
    with col:
        st.markdown(
            f"""
            <div class="forecast-card" style="background:{bg_c};">
                <p class="forecast-day" style="color:{txt_c};">{label}</p>
                <p class="forecast-number" style="color:{txt_c}; font-size:{size};">{value:.0f}</p>
                <p class="forecast-cat" style="color:{txt_c};">{cat}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

max_forecast = max(pred_24h, pred_48h, pred_72h)
if max_forecast > 150:
    st.markdown(
        f"""
        <div class="alert-banner">
            <span style="font-size:1.8rem;">🚨</span>
            <div>
                <p style="margin:0; font-weight:800; color:white; font-family:'Baloo 2',sans-serif; font-size:1.05rem;">Hazardous Air Quality Alert</p>
                <p style="margin:0; font-size:0.85rem; color:#ffe8e8; font-family:'Nunito',sans-serif; font-weight:600;">AQI is expected to reach unhealthy levels within 3 days. Limit outdoor activity.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<p class="section-label" style="margin-top:32px;">📈 Trend: Last 7 Days + Forecast</p>', unsafe_allow_html=True)

recent = df.tail(7)[["timestamp", "pm25"]].copy()
last_time = df["timestamp"].max()
forecast_points = pd.DataFrame({
    "timestamp": [
        last_time,
        last_time + pd.Timedelta(hours=24),
        last_time + pd.Timedelta(hours=48),
        last_time + pd.Timedelta(hours=72)
    ],
    "pm25": [current_pm25, pred_24h, pred_48h, pred_72h],
})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent["timestamp"], y=recent["pm25"],
    mode="lines+markers", name="Historical",
    line=dict(color="#4d9de0", width=4, shape="spline"),
    marker=dict(size=9, color="#4d9de0")
))
fig.add_trace(go.Scatter(
    x=forecast_points["timestamp"], y=forecast_points["pm25"],
    mode="lines+markers", name="Forecast",
    line=dict(color="#ff9f4a", width=4, dash="dot", shape="spline"),
    marker=dict(size=9, color="#ff9f4a")
))
fig.update_layout(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    xaxis_title=None,
    yaxis_title="PM2.5",
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    font=dict(family="Nunito", color="#2d2a26", size=13),
    xaxis=dict(showgrid=False, color="#2d2a26"),
    yaxis=dict(gridcolor="#f0ebe4", color="#2d2a26"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#2d2a26", size=13))
)

with st.container(key="chart-card"):
    st.plotly_chart(fig, use_container_width=True, key="aqi_trend_chart")