# CompetitorLens v4 — Price Finder (NYC Filters + Unit Switch + Map)

**What’s new**
- Geography filter: choose **NYC Five Boroughs**, **New York State**, or **Custom location**
- Borough multi-select + map preview (borough centroids)
- **Unit switch**: default **sq ft**, or **m²**; we normalize to USD per m²/month and display per sq ft too

**Deploy**
- Entry: `competitorlens.py`
- `pip install -r requirements.txt`
- Secrets (optional for LLM refine / search boosters):
```
# OPENAI_API_KEY = "sk-..."
# BRAVE_API_KEY = "your-brave-key"
# SERPAPI_API_KEY = "your-serpapi-key"
```

**Tips**
- Some providers gate prices behind ZIP/date; this app focuses on public pricing and PDFs.
- Tighten geography by selecting fewer boroughs or using a precise **Custom location** string.
