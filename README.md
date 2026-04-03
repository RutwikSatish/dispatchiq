# DispatchIQ — Last-Mile Delivery Defect Detection & Cost Analytics

**Tools:** Python · pandas · Streamlit · Plotly · SQLite · Groq AI

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What this solves

In high-volume last-mile networks, delivery defects (mis-sorts, delays) cluster around
specific carrier × zone × hour combinations that aggregate OTD metrics miss entirely.
DispatchIQ surfaces those clusters, models the cost impact, and simulates carrier
reallocation experiments before network changes are committed.

## Features

| Tab | What it shows |
|-----|--------------|
| Overview | OTD % by carrier, defect rate by zone, weekly quality/cost trends |
| Defect Analysis | Carrier × zone heatmap, defect type breakdown, hour-of-day patterns, SQL query of top 23 high-defect segments |
| Cost Analytics | Quality-cost scatter, redelivery cost by carrier, top 4 under-performing pairs, AI executive brief |
| Experiment Planner | A/B framework for carrier reallocation, pre/post defect rate modelling, top-5 reallocation impact |

## Resume claims validated by this app

- **85,000+ delivery records** — synthetic, 90-day window, 6 zones, 8 carriers
- **23 high-defect segments** — carrier × zone pairs with defect rate ≥ 2× network average
- **2.4× above network average** — worst-case cluster ratio surfaced in Defect Analysis tab
- **$340K in avoidable redelivery cost** — driven by top 4 carrier-zone pairings (Cost tab)
- **18% defect reduction modeled** — from top-5 carrier reallocations (Experiment Planner tab)
- **A/B experiment framework** — pre/post comparison with 50/50 treatment split

## Groq AI

Add a Groq API key in the sidebar (`gsk_...`) to generate AI-powered network cost briefs
in the Cost Analytics tab. Free key at [console.groq.com](https://console.groq.com).

Add to `.streamlit/secrets.toml` to pre-load:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Then update `app.py` sidebar section:
```python
_key = st.secrets.get("GROQ_API_KEY", "")
groq_key = _key or st.text_input("Groq API Key (optional)", type="password")
```
