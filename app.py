

"""
app.py — Agricultural Intelligence Network (main Streamlit web app)
 
Run locally:    streamlit run app.py
Deploy:          push to GitHub, connect at share.streamlit.io (see README)
 
Tabs:
  1. Get Advisory     — location + crop -> AI-generated localized advisory
  2. Diagnose Disease  — upload a leaf photo -> CNN classification + treatment
  3. Cross-State View  — the "cooperation / digital public good" dashboard
"""
 
import streamlit as st
import pandas as pd
import tempfile, os
 
# Pull API keys from Streamlit secrets (if configured) into env vars, so
# data_utils.py / advisory_engine.py (which read os.environ) pick them up
# whether running locally with .streamlit/secrets.toml or on Streamlit Cloud.
try:
    for key in ("GEMINI_API_KEY", "OPENWEATHER_API_KEY"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # no secrets.toml configured — app still runs with simulated data
 
from data_utils import get_soil_health, get_weather_forecast, get_ndvi_soil_moisture, get_available_locations
from advisory_engine import generate_advisory
from disease_model import predict as predict_disease
 
st.set_page_config(page_title="Agri Intelligence Network", page_icon="🌾", layout="wide")
 
st.title("🌾 Agricultural Intelligence Network")
st.caption("AI-powered, satellite-informed advisories for smallholder farmers — designed as a shared digital public good across states.")
 
# ------------------------------------------------------------------
# SIDEBAR: DATA SOURCE STATUS
# Shows judges at a glance which data sources are real vs. simulated,
# instead of relying on you explaining it verbally.
# ------------------------------------------------------------------
_real_satellite_path = os.path.join(os.path.dirname(__file__), "sample_data", "real_satellite_sample.csv")
 
with st.sidebar:
    st.markdown("### 📡 Data Source Status")
    st.caption("Live status of each data feed powering this demo.")
 
    def _status_row(label, is_real, real_note, sim_note):
        icon = "🟢" if is_real else "🟡"
        state = "Real" if is_real else "Simulated"
        note = real_note if is_real else sim_note
        st.markdown(f"{icon} **{label}:** {state}")
        st.caption(note)
 
    _status_row(
        "Soil health", True,
        "Soil Health Card sample dataset (12 districts)",
        "",
    )
    _status_row(
        "Weather", bool(os.environ.get("OPENWEATHER_API_KEY")),
        "Live OpenWeatherMap forecast",
        "Realistic synthetic forecast (no API key set)",
    )
    _status_row(
        "Satellite (NDVI / soil moisture)", os.path.exists(_real_satellite_path),
        "Real Sentinel-2 + ERA5-Land data (pulled via Earth Engine)",
        "Simulated using realistic crop-growth curves",
    )
    _status_row(
        "Advisory engine", bool(os.environ.get("GEMINI_API_KEY")),
        "Gemini AI-generated (falls back to rule-based if unavailable)",
        "Rule-based expert system (transparent, auditable, offline)",
    )
    st.markdown("---")
    st.caption("🟢 = live/real data · 🟡 = simulated for this demo")
 
locations_df = get_available_locations()
 
tab1, tab2, tab3 = st.tabs(["📋 Get Advisory", "🔬 Diagnose Disease", "🤝 Cross-State Cooperation"])
 
# ------------------------------------------------------------------
# TAB 1: ADVISORY
# ------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.4])
 
    with col1:
        st.subheader("Your field")
        state = st.selectbox("State", sorted(locations_df["state"].unique()))
        districts = locations_df[locations_df["state"] == state]["district"].unique()
        district = st.selectbox("District", sorted(districts))
        crop = st.selectbox("Crop", ["Wheat", "Cotton", "Paddy (Rice)", "Tomato", "Potato", "Maize", "Chickpea (Gram)"])
        growth_days = st.slider("Days since sowing", 1, 150, 45)
        language = st.selectbox("Advisory language", ["English", "Hindi", "Marathi", "Bhojpuri"])
        run = st.button("Generate Advisory", type="primary", use_container_width=True)
 
    if run:
        loc_row = locations_df[(locations_df["state"] == state) & (locations_df["district"] == district)].iloc[0]
        lat, lon = float(loc_row["lat"]), float(loc_row["lon"])
 
        with st.status("Gathering field data...", expanded=True) as status:
            st.write("🌱 Fetching soil health data...")
            soil = get_soil_health(state, district)
            st.write("🌦️ Fetching weather forecast...")
            weather = get_weather_forecast(lat, lon)
            st.write("🛰️ Fetching satellite / NDVI data...")
            ndvi_data = get_ndvi_soil_moisture(lat, lon, growth_days)
            st.write("🧠 Generating advisory...")
            advisory = generate_advisory(crop, growth_days, soil, weather, ndvi_data, language=language)
            status.update(label="Advisory ready", state="complete", expanded=False)
 
        with col2:
            st.subheader(advisory["headline"])
 
            conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(advisory["confidence"], "⚪")
            st.markdown(f"**Confidence:** {conf_color} {advisory['confidence']} — _{advisory['confidence_reason']}_")
 
            st.success(f"**Recommended action:** {advisory['recommendation']}")
            st.markdown(f"**Why:** {advisory['reasoning']}")
 
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Regenerative Score", f"{advisory['regenerative_score']}/10")
            c2.caption(advisory["regenerative_note"])
 
            with st.expander("More tips"):
                for tip in advisory.get("secondary_tips", []):
                    st.markdown(f"- {tip}")
 
            with st.expander("Raw data used for this advisory (transparency)"):
                st.json({"soil": soil, "weather": weather, "satellite": ndvi_data})
 
# ------------------------------------------------------------------
# TAB 2: DISEASE DIAGNOSIS
# ------------------------------------------------------------------
with tab2:
    st.subheader("Upload a photo of the affected leaf")
    uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
 
    if uploaded:
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)
 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
 
        with st.spinner("Analyzing leaf..."):
            result = predict_disease(tmp_path)
        os.unlink(tmp_path)
 
        with col2:
            if result.get("note"):
                st.warning(result["note"])
            label = result["predicted_class"].replace("___", " — ").replace("_", " ")
            st.subheader(label)
            st.progress(min(result["confidence"], 1.0))
            st.caption(f"Confidence: {result['confidence']*100:.1f}%")
            st.info(f"**Suggested treatment:** {result['treatment']}")
    else:
        st.caption("Tip: for the demo, download a few sample images from the PlantVillage dataset ahead of time so you're not relying on live wifi to find test images.")
 
# ------------------------------------------------------------------
# TAB 3: CROSS-STATE COOPERATION DASHBOARD
# ------------------------------------------------------------------
with tab3:
    st.subheader("Shared Agricultural Data Layer")
    st.caption(
        "This is the 'digital public good' layer: states publish anonymized, "
        "aggregated field metrics through a common schema so patterns and "
        "advisories can be compared and reused across state lines — without "
        "sharing raw farmer-level data."
    )
 
    st.markdown("#### District coverage map")
    map_df = locations_df.copy()
    # Give each state a distinct colour so the map reads as "two states
    # cooperating," not just a scatter of identical dots.
    state_colors = {
        "Maharashtra": [255, 140, 0],   # orange
        "Bihar": [0, 128, 0],           # green
    }
    map_df["color"] = map_df["state"].apply(lambda s: state_colors.get(s, [70, 130, 180]))
    try:
        st.map(map_df, latitude="lat", longitude="lon", color="color", size=2000)
    except TypeError:
        # Older Streamlit versions don't support the color/size kwargs —
        # fall back to a plain map so this never breaks the demo.
        st.map(map_df[["lat", "lon"]])
    st.caption("🟠 Maharashtra districts · 🟢 Bihar districts")
 
    rows = []
    for _, loc in locations_df.iterrows():
        soil = get_soil_health(loc["state"], loc["district"])
        ndvi_data = get_ndvi_soil_moisture(loc["lat"], loc["lon"], 45)
        rows.append({
            "State": loc["state"],
            "District": loc["district"],
            "Avg NDVI": ndvi_data["ndvi"],
            "Soil Moisture %": ndvi_data["soil_moisture_pct"],
            "Nitrogen (kg/ha)": soil["nitrogen_kg_ha"],
            "Organic Carbon %": soil["organic_carbon_pct"],
            "Soil Type": soil["soil_type"],
        })
    df = pd.DataFrame(rows)
 
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    st.markdown("#### Cross-state comparison")
    metric = st.selectbox("Compare states by:", ["Avg NDVI", "Soil Moisture %", "Nitrogen (kg/ha)", "Organic Carbon %"])
    chart_df = df.groupby("State")[metric].mean().reset_index()
    st.bar_chart(chart_df, x="State", y=metric)
 
    st.markdown("---")
    st.markdown(
        "**Shared schema (this is what would let real state agri-departments "
        "plug into a common API):**"
    )
    st.code(
        """{
  "region_id": "state.district",
  "timestamp": "ISO8601",
  "soil": {"n_kg_ha": float, "p_kg_ha": float, "k_kg_ha": float, "ph": float, "organic_carbon_pct": float},
  "satellite": {"ndvi": float, "soil_moisture_pct": float},
  "weather": {"rain_next_5d_mm": float, "avg_temp_c": float},
  "advisory_summary": {"top_recommendation": str, "regenerative_score": int}
}""",
        language="json",
    )
 
st.markdown("---")
st.caption("Built for [Hackathon Name] · Theme: Cooperation · Data sources: Soil Health Card (sample), OpenWeatherMap, Sentinel-2 NDVI (simulated in demo mode)")
 



