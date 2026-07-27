import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(
    page_title="AI National Supply Chain Risk Intelligence Platform", layout="wide")
st.sidebar.title("🚚 AI Supply Chain Platform")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Dashboard", "🤖 Prediction", "📈 Analytics"]
)

if page == "🏠 Home":
    st.title("🚚 AI National Supply Chain Risk Intelligence Platform")

    st.markdown("""
    ### 🎯 Project Objective

    This AI platform predicts supply chain risks, analyzes shipment performance,
    and helps businesses reduce delivery delays using Machine Learning.

    ### 👨‍💻 Features
    ✅ AI Risk Prediction
    ✅ Interactive Dashboard
    ✅ Supply Chain Analytics
    ✅ Business Insights

    > ⚠️ **Demo mode**: this app is running on randomly generated sample data
    > and a dummy predictor, for UI testing only — no real dataset or model
    > file is used.
    """)

# ---------------------------------------------------------------------------
# Dummy in-memory dataset (replaces reading Cleaned_SupplyChain_Dataset.csv)
# ---------------------------------------------------------------------------
@st.cache_data
def make_dummy_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    states = ["California", "Texas", "New York", "Florida", "Illinois",
              "Ohio", "Georgia", "Washington", "Arizona", "Colorado"]

    df = pd.DataFrame({
        "Order State": rng.choice(states, size=n),
        "Sales per customer": rng.uniform(20, 1200, size=n).round(2),
        "Benefit per order": rng.uniform(-50, 400, size=n).round(2),
        "Days for shipment (scheduled)": rng.integers(1, 10, size=n),
        "Order Item Quantity": rng.integers(1, 20, size=n),
        "Order Item Product Price": rng.uniform(10, 500, size=n).round(2),
        "Late_delivery_risk": rng.integers(0, 2, size=n),
        # roughly within the continental US for the map demo
        "Latitude": rng.uniform(25, 49, size=n).round(4),
        "Longitude": rng.uniform(-124, -67, size=n).round(4),
    })
    return df

df = make_dummy_data()

st.sidebar.header("🔍 Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["Order State"].dropna().unique().tolist())
)

if selected_state != "All":
    filtered_df = df[df["Order State"] == selected_state]
else:
    filtered_df = df

st.write("### Supply Chain Risk Dashboard")

high_risk = len(filtered_df[filtered_df["Late_delivery_risk"] == 1])
st.metric("🚨 High Risk Shipments", high_risk)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Orders", len(filtered_df))
col2.metric("💰 Total Sales", f"{filtered_df['Sales per customer'].sum():,.0f}")
col3.metric("📈 Average Benefit", f"{filtered_df['Benefit per order'].mean():.2f}")
col4.metric("🚚 Average Shipping Days", f"{filtered_df['Days for shipment (scheduled)'].mean():.1f}")

if page == "📊 Dashboard":
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head())

    st.subheader("Dataset Information")
    st.write(f"Rows: {filtered_df.shape[0]}")
    st.write(f"Columns: {filtered_df.shape[1]}")

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV Report",
        csv,
        "Supply_Chain_Report.csv",
        "text/csv"
    )

# ---------------------------------------------------------------------------
# Dummy predictor (replaces joblib.load("supply_chain_risk_model.pkl"))
# ---------------------------------------------------------------------------
def dummy_predict(benefit, sales, quantity, price, days):
    """Simple rule-of-thumb stand-in for a trained model, just for UI testing."""
    score = 0
    if days > 5:
        score += 1
    if quantity > 10:
        score += 1
    if sales > 500:
        score += 1
    if benefit < 100:
        score += 1
    return 1 if score >= 2 else 0

if page == "🤖 Prediction":
    st.subheader("🔮 Supply Chain Risk Prediction")
    st.caption("Demo mode: using a simple rule-based stand-in, not a trained model.")

    benefit = st.number_input("Benefit per Order", value=150.0)
    sales = st.number_input("Sales per Customer", value=300.0)
    quantity = st.number_input("Order Item Quantity", value=5)
    price = st.number_input("Order Item Product Price", value=100.0)
    days = st.number_input("Days for Shipment (Scheduled)", value=3)

    if st.button("Predict Risk"):
        prediction = dummy_predict(benefit, sales, quantity, price, days)

        if prediction == 1:
            st.error("🚨 High Risk Shipment")

            st.subheader("🤖 AI Recommendation")
            st.write("🚚 Switch to faster shipping mode")
            st.write("🏭 Evaluate supplier performance")
            st.write("📦 Increase safety stock level")
            st.write("📊 Monitor high-risk regions")

            st.subheader("🧠 Why is this shipment High Risk?")
            if days > 5:
                st.write("📅 Scheduled shipping time is high.")
            if quantity > 10:
                st.write("📦 Large order quantity increases delivery risk.")
            if sales > 500:
                st.write("💰 High sales volume requires faster logistics.")
            if benefit < 100:
                st.write("📉 Low benefit per order may reduce supply chain efficiency.")

        else:
            st.success("✅ Low Risk Shipment")

            st.subheader("🤖 AI Recommendation")
            st.write("✅ Supply chain operations are stable")
            st.write("📈 Continue current strategy")
            st.write("🔍 Maintain supplier monitoring")

            st.subheader("🧠 Why is this shipment Low Risk?")
            st.write("✅ Delivery schedule is within normal limits.")
            st.write("✅ Supply chain conditions appear stable.")
            st.write("✅ No major operational risk detected.")

if page == "📈 Analytics":
    st.subheader("📊 Sales Distribution")
    st.bar_chart(filtered_df["Sales per customer"].head(20))

    st.subheader("📈 Benefit Distribution")
    st.line_chart(filtered_df["Benefit per order"].head(20))

    st.subheader("📊 Risk Distribution")
    risk_count = filtered_df["Late_delivery_risk"].value_counts()
    st.pyplot(
        risk_count.plot.pie(
            autopct="%1.1f%%",
            figsize=(5, 5)
        ).get_figure()
    )

    st.subheader("🗺️ State-wise Risk Analysis")
    state_risk = filtered_df.groupby("Order State")["Benefit per order"].mean()
    st.bar_chart(state_risk)

    st.subheader("🌍 Interactive Risk Map")
    map_data = filtered_df[["Latitude", "Longitude"]].dropna().head(500)

    if not map_data.empty:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=map_data["Latitude"].mean(),
                    longitude=map_data["Longitude"].mean(),
                    zoom=2,
                    pitch=40,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_data,
                        get_position='[Longitude, Latitude]',
                        get_radius=30000,
                        get_fill_color='[255, 0, 0, 160]',
                        pickable=True,
                    ),
                ],
            )
        )
    else:
        st.info("No latitude/longitude data available for the current filter.")

    st.subheader("🌍 Live Risk Intelligence")
    alerts = [
        "⚠ Heavy rainfall may affect deliveries in South India.",
        "🚢 Port congestion reported in Singapore.",
        "⛽ Fuel price increase may impact transportation cost.",
        "🚛 Highway traffic delays expected in North Region."
    ]
    for alert in alerts:
        st.warning(alert)
