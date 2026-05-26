# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="EcoPlate AI",
    page_icon="🍽️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

body {
    background-color: #050816;
}

.stApp {
    background: linear-gradient(to right, #050816, #091120);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #070d1d;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Remove default radio look */
div[role="radiogroup"] label {
    background-color: #0f172a;
    padding: 12px;
    margin-bottom: 10px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    transition: 0.3s;
}

div[role="radiogroup"] label:hover {
    background: linear-gradient(to right, #00c85322, #00e67622);
    border: 1px solid #00e676;
}

/* Main headings */
.main-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
}

.green-text {
    color: #00e676;
}

/* Hero Box */
.hero-box {
    background: linear-gradient(135deg, #07111f, #0a192f);
    padding: 40px;
    border-radius: 25px;
    border: 1px solid #1f2937;
    margin-bottom: 25px;
    box-shadow: 0px 0px 20px rgba(0,255,170,0.1);
}

/* Cards */
.card {
    background: #0b1220;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #1f2937;
    box-shadow: 0px 0px 15px rgba(0,255,170,0.05);
    margin-bottom: 20px;
}

.card:hover {
    border: 1px solid #00e676;
    transform: scale(1.01);
    transition: 0.3s;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg,#081221,#0d1b2a);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #1f2937;
}

/* Remove Streamlit top space */
.block-container {
    padding-top: 1rem;
}

/* Slider */
.stSlider > div > div {
    color: #00e676;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right,#00c853,#00e676);
    color: black;
    border: none;
    border-radius: 10px;
    font-weight: bold;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #0b1220;
}

/* Success box */
.stSuccess {
    background-color: #072b1d;
}

/* Info box */
.stInfo {
    background-color: #0b2447;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATASET ----------------

np.random.seed(42)

days = pd.date_range(start="2025-01-01", periods=120)

data = pd.DataFrame({
    "Date": days,
    "Orders": np.random.randint(80, 300, 120),
    "Leftover_Food_kg": np.random.randint(5, 50, 120),
    "Peak_Hour": np.random.choice(["Lunch", "Dinner", "Evening"], 120),
    "Season": np.random.choice(["Winter", "Summer", "Monsoon"], 120),
    "Inventory_kg": np.random.randint(50, 200, 120)
})

data["Demand"] = (
    data["Orders"] * 0.75 +
    np.random.randint(10, 40, 120)
).astype(int)

# ---------------- MACHINE LEARNING ----------------

X = data[["Orders", "Inventory_kg", "Leftover_Food_kg"]]
y = data["Demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<div style='padding-top:20px; padding-bottom:20px;'>

<h1 style='color:white;'>🍃 EcoPlate AI</h1>

<p style='color:#9ca3af;'>
Smart Restaurant Analytics
</p>

</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🗑️ Waste Analysis",
        "🤖 Demand Prediction",
        "📦 Inventory Optimization",
        "📉 Waste Prediction Dashboard"
    ]
)

# ---------------- HOME ----------------

if page == "🏠 Home":

    st.markdown("""
    <div class='hero-box'>

    <p style='color:#00e676;font-size:18px;'>
    AI Powered Restaurant Analytics
    </p>

    <div class='main-title'>
    Food Waste <span class='green-text'>Reduction Analytics</span>
    </div>

    <br>

    <p style='font-size:20px;color:#d1d5db;'>
    Smarter Decisions. Less Waste. More Profit.
    </p>

    <p style='font-size:18px;color:#9ca3af;line-height:1.8;'>
    An intelligent analytics platform that helps restaurants predict demand,
    reduce food waste, optimize inventory, and improve operational efficiency.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🎯 Problem Statement")

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        <div class='card'>

        Restaurants waste huge amounts of food daily due to:

        <br><br>

        🔴 Overproduction<br><br>
        🟠 Inaccurate demand forecasting<br><br>
        🔵 Seasonal fluctuations<br><br>
        🟣 Poor inventory management

        <br><br>

        This increases operational costs and impacts sustainability.

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class='metric-card'>

        <h1 style='color:#00e676;'>40%</h1>

        <p>Waste Reduction Goal</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 📌 Quick Insights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
        <h1 style='color:#00e676;'>{int(data["Orders"].mean())}</h1>
        <p>Average Daily Orders</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
        <h1 style='color:#00e676;'>{round(data["Leftover_Food_kg"].mean(),1)} kg</h1>
        <p>Average Food Waste</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
        <h1 style='color:#00e676;'>{round(100-mae,1)}%</h1>
        <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- DATASET ANALYSIS ----------------

elif page == "📊 Dataset Analysis":

    st.markdown("""
    <div class='hero-box'>
    <h1>📊 Dataset Analysis</h1>
    <p style='color:#9ca3af;'>
    Analyze restaurant operational data and food demand patterns.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(data, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.histogram(
            data,
            x="Orders",
            nbins=20,
            title="Orders Distribution"
        )

        fig1.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.pie(
            data,
            names="Season",
            title="Season Distribution"
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1220"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ---------------- WASTE ANALYSIS ----------------

elif page == "🗑️ Waste Analysis":

    st.markdown("""
    <div class='hero-box'>
    <h1>🗑️ Waste Analysis</h1>
    <p style='color:#9ca3af;'>
    Discover food waste trends and identify high waste periods.
    </p>
    </div>
    """, unsafe_allow_html=True)

    fig3 = px.line(
        data,
        x="Date",
        y="Leftover_Food_kg",
        title="Daily Food Waste Trend"
    )

    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
    )

    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.bar(
        data,
        x="Peak_Hour",
        y="Leftover_Food_kg",
        color="Peak_Hour",
        title="Food Waste by Peak Hour"
    )

    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
    )

    st.plotly_chart(fig4, use_container_width=True)

# ---------------- DEMAND PREDICTION ----------------

elif page == "🤖 Demand Prediction":

    st.markdown("""
    <div class='hero-box'>
    <h1>🤖 Demand Prediction</h1>
    <p style='color:#9ca3af;'>
    Predict restaurant food demand using machine learning.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        orders = st.slider("Daily Orders", 50, 400, 150)

    with col2:
        inventory = st.slider("Inventory (kg)", 20, 300, 100)

    with col3:
        leftover = st.slider("Leftover Food (kg)", 1, 60, 10)

    input_data = np.array([[orders, inventory, leftover]])

    predicted_demand = model.predict(input_data)[0]

    st.markdown(f"""
    <div class='card'>

    <h1 style='color:#00e676;'>
    Predicted Demand: {int(predicted_demand)} Meals
    </h1>

    </div>
    """, unsafe_allow_html=True)

    st.info(f"Model Accuracy Score: {round(100-mae,2)}%")

# ---------------- INVENTORY ----------------

elif page == "📦 Inventory Optimization":

    st.markdown("""
    <div class='hero-box'>
    <h1>📦 Inventory Optimization</h1>
    <p style='color:#9ca3af;'>
    Optimize inventory and reduce unnecessary stock waste.
    </p>
    </div>
    """, unsafe_allow_html=True)

    data["Suggested_Inventory"] = data["Demand"] + 10

    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Inventory_kg"],
        mode='lines',
        name='Current Inventory'
    ))

    fig5.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Suggested_Inventory"],
        mode='lines',
        name='Suggested Inventory'
    ))

    fig5.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        title="Inventory Optimization Analysis"
    )

    st.plotly_chart(fig5, use_container_width=True)

# ---------------- DASHBOARD ----------------

elif page == "📉 Waste Prediction Dashboard":

    st.markdown("""
    <div class='hero-box'>
    <h1>📉 Waste Prediction Dashboard</h1>
    <p style='color:#9ca3af;'>
    AI dashboard showing demand trends and food waste analytics.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
        <h1 style='color:#00e676;'>{data["Orders"].sum()}</h1>
        <p>Total Orders</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
        <h1 style='color:#00e676;'>{data["Leftover_Food_kg"].sum()} kg</h1>
        <p>Total Food Waste</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
        <h1 style='color:#00e676;'>{int(data["Demand"].mean())}</h1>
        <p>Average Demand</p>
        </div>
        """, unsafe_allow_html=True)

    heatmap_data = data.pivot_table(
        values="Leftover_Food_kg",
        index="Season",
        columns="Peak_Hour",
        aggfunc="mean"
    )

    fig6 = px.imshow(
        heatmap_data,
        text_auto=True,
        title="Food Waste Heatmap"
    )

    fig6.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220"
    )

    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.scatter(
        data,
        x="Orders",
        y="Leftover_Food_kg",
        color="Season",
        size="Demand",
        title="Orders vs Waste"
    )

    fig7.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
    )

    st.plotly_chart(fig7, use_container_width=True)