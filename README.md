# 🌾 Agricultural Intelligence Network
### Theme: Cooperation — AI-powered, cross-state digital agriculture platform

This repo is a working, deployable starting point. Everything runs out of the
box in **simulated data mode** (no API keys needed) so you can demo it
immediately, then swap in real API keys and a trained model as you go.

---

## 0. What's already built for you

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit web app (3 tabs: Advisory, Disease Diagnosis, Cross-State Dashboard) |
| `data_utils.py` | Soil / weather / NDVI data fetchers, with automatic fallback to realistic simulated data |
| `advisory_engine.py` | Claude-powered advisory generator (confidence-aware, regenerative scoring) with a rule-based fallback |
| `disease_model.py` | MobileNetV2 transfer-learning training script + inference function |
| `sample_data/soil_health_sample.csv` | Sample Soil Health Card data across 2 states / 6 districts |
| `requirements.txt` | All Python dependencies |

**This already runs.** Try it right now:
```bash
pip install -r requirements.txt
streamlit run app.py
```
It'll work immediately using simulated weather/NDVI data and a rule-based
advisory fallback — meaning you have a demoable product on hour one, and every
day after this is about making it *more real* and *more polished*, not about
getting something on screen for the first time.

---

## DAY 1 — Real data + trained model

### Step 1: Get your API keys (30 min)
- **Anthropic API key**: console.anthropic.com → get a key → this powers the advisory text
- **OpenWeatherMap key** (optional but easy, free): openweathermap.org/api
- Create `.streamlit/secrets.toml` (copy from `secrets.toml.example`) and fill in your keys:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENWEATHER_API_KEY = "..."
```
- Restart `streamlit run app.py` — Tab 1 will now generate real AI advisories instead of the rule-based fallback.

### Step 2: Expand the soil dataset (1 hr)
- Download real Soil Health Card data for your chosen 2 states: https://soilhealth.dac.gov.in/
- Add more districts to `sample_data/soil_health_sample.csv` in the same format (state, district, lat, lon, N, P, K, pH, organic carbon, soil type)
- Pull lat/lon per district from Google Maps if not in the source data

### Step 3: Train the disease classifier (3-4 hrs, do this on Colab/Kaggle for free GPU)
1. Download the PlantVillage dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Unzip so you get one folder per class (this is the default layout)
3. On Colab/Kaggle (free T4 GPU), upload `disease_model.py` and the dataset, then run:
   ```bash
   python disease_model.py --train --data_dir ./plantvillage_data --epochs 5
   ```
4. This freezes a pretrained MobileNetV2 backbone and only trains a new
   classification head — expect **85-95% validation accuracy within 5 epochs**
   because PlantVillage is a clean, well-separated dataset and transfer
   learning does most of the work for you.
5. Download the two output files back to your project: `models/disease_model.h5` and `models/class_indices.json`
6. Once present, `app.py` Tab 2 automatically switches from stub predictions to real ones — no code change needed.
7. **Extend `TREATMENT_GUIDE`** in `disease_model.py` to cover every class name your trained model actually outputs (check `class_indices.json` for the exact list) — the demo only ships a partial dictionary.

### Step 4 (real satellite data — optional, do only if time allows)
- Sign up for Google Earth Engine (free): https://earthengine.google.com/
- Replace the body of `get_ndvi_soil_moisture()` in `data_utils.py` with the real `ee.Initialize()` call shown in the docstring comment
- **If you skip this**: that's fine — say so honestly in your pitch ("NDVI is simulated for the demo using realistic crop-growth curves; the architecture is built to swap in live Sentinel-2 data via Earth Engine with no other code changes"). Judges respect honesty about scope far more than a demo that pretends fake data is real.

---

## DAY 2 — Polish the product + build the cooperation story

### Step 1: Walk through the full user flow yourself
- Go through all 3 tabs as if you were a judge. Note anything confusing.
- Try at least 3 different district/crop combos in Tab 1 — make sure advisories actually differ meaningfully based on the data (not generic text every time).

### Step 2: Strengthen Tab 3 (Cross-State Cooperation) — this is your theme differentiator
- This tab is intentionally the one most tied to the "cooperation" theme in your problem statement. Spend real time here, not just Tabs 1-2.
- Consider adding: a short paragraph explaining *why* the shared schema approach protects farmer privacy while still enabling state-level cooperation (anonymized/aggregated, not raw records) — this directly answers "digital public good" from the brief.
- If you have time, add a second bar chart comparing organic carbon trends — ties into "regenerative."

### Step 3: Add the vernacular / accessibility angle
- Tab 1 already has a language selector wired into the Claude prompt — test it with Hindi/Marathi and confirm the output actually comes back in that language.
- In your pitch, explicitly call out: most target users won't be comfortable in English-only tools — this is a deliberate accessibility choice, not a nice-to-have.

### Step 4: Error-proof it
- Test the app with no internet (simulated mode) — does it still work? It should.
- Test with a broken/huge image upload in Tab 2 — does it crash? Add a `try/except` around `predict_disease()` in `app.py` if needed.

---

## DAY 3 — Deploy + pitch

### Step 1: Push to GitHub
```bash
cd agri-intelligence
git init
git add .
echo ".streamlit/secrets.toml" >> .gitignore
echo "models/*.h5" >> .gitignore   # optional: model files can be large; Streamlit Cloud can handle up to ~1GB but keep repo lean if possible
git commit -m "Agri Intelligence Network — hackathon submission"
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud (free, ~5 min)
1. Go to https://share.streamlit.io → "New app"
2. Connect your GitHub repo, select `app.py` as the entry point
3. In **Advanced settings → Secrets**, paste the same contents as your local `.streamlit/secrets.toml`
4. Deploy — you'll get a public URL like `https://your-app.streamlit.app`
5. **Test the live URL yourself** before presenting, at least twice, including on a phone browser

### Step 3: Record a backup demo video (non-negotiable — do this)
- Screen-record a full walkthrough of all 3 tabs, 2-3 minutes, narrated
- Venue wifi fails at hackathons constantly — a working video backup has saved countless teams

### Step 4: Prep your pitch narrative
Structure (2-3 min):
1. **Problem** (20s): smallholder farmers lack data-driven guidance; states don't share agri-data → repeated failures, no collective learning
2. **Demo** (90s): live walkthrough — advisory → disease diagnosis → cross-state dashboard
3. **What makes this a "digital public good"** (20s): open schema, any state agri-department could plug in; anonymized/aggregated not raw data
4. **Honest scope statement** (20s): "For this build we used simulated satellite data with the exact architecture to plug in live Sentinel-2/Earth Engine data — the interoperability layer and advisory engine are fully real and working."
5. **What's next** (10s): SMS/USSD fallback for zero-smartphone farmers, more states, more crops

---

## Architecture recap (for your slide deck)

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ Soil Health Card │    │ Weather API       │    │ Satellite NDVI      │
│ (govt open data) │    │ (OpenWeatherMap)  │    │ (Sentinel-2 / GEE)  │
└────────┬─────────┘    └────────┬──────────┘    └──────────┬──────────┘
         └───────────────────────┼──────────────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │  Advisory Engine (Claude) │───► Confidence-aware,
                    │  + regenerative scoring   │     regenerative-scored,
                    └─────────────┬─────────────┘     vernacular advisory
                                  │
                    ┌─────────────▼─────────────┐
         Photo ───► │ Disease Classifier (CNN)   │───► Diagnosis + treatment
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Shared State Data Layer /   │───► Cross-state cooperation
                    │ Common JSON Schema           │     dashboard
                    └───────────────────────────┘
```

## Innovation angles you can lean on in the pitch
1. **Confidence-aware advisories** — the AI explicitly states how sure it is and why, instead of always sounding authoritative (real ML maturity signal)
2. **Regenerative scoring** — every recommendation is scored 0-10 on long-term soil health, not just short-term yield
3. **Federated cooperation model** — states share anonymized aggregates through a common schema, sidestepping real data-privacy issues while still enabling cross-state learning
4. **Vernacular-first** — advisory language selector (Hindi/Marathi/Bhojpuri etc.), because English-only tools exclude most of the target user base
5. **Graceful degradation** — every data source has a transparent simulated fallback, so the platform is honest about live vs. simulated data instead of silently failing or faking certainty

## Known limitations to state upfront (don't let a judge "catch" these — say them yourself)
- NDVI/soil moisture is simulated for the hackathon build (real Earth Engine integration is a drop-in swap, documented in `data_utils.py`)
- Disease classifier is trained on PlantVillage (leaf-level images under relatively clean conditions) — real field photos in poor lighting will need more training data to reach the same accuracy
- Only 2 states / 6 districts of sample data — proves the concept, not yet the scale
