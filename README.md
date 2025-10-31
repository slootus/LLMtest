# CompetitorLens v6 — Minimal (single search field, Google-first)

Type a natural request (e.g. “Find the pricing of mini warehouses in New York for 10m2 warehouses”).  
The app will:
1) Parse the size & business type from your text (m² / sq ft / 10×10 ft, etc.)
2) Search Google first (SerpAPI / Google Programmable Search), with DDG/Brave fallbacks
3) Crawl likely pricing/size pages
4) Extract prices and normalize to **USD per m² per month** (+ per sq ft)
5) Show a table + trimmed-average and download buttons

## Secrets (Streamlit → Settings → Secrets)
```
# Prefer one of:
SERPAPI_API_KEY = "your-serpapi-key"
# or
# GOOGLE_CSE_API_KEY = "your-google-api-key"
# GOOGLE_CSE_CX = "your-cse-cx-id"

# Optional:
# OPENAI_API_KEY = "sk-..."   # only for LLM refine
# BRAVE_API_KEY = "your-brave-key"
```

## Run
```
pip install -r requirements.txt
streamlit run competitorlens.py
```
