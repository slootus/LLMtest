# CompetitorLens (Streamlit)

Prompt-driven competitor & product research:
- Enter a prompt (e.g., "Compare olive oil exporters in France by range, certifications, and price")
- App searches + crawls relevant sites
- LLM extracts structured data
- You get a comparison table + executive summary

## Deploy on Streamlit Community Cloud
1) Create a new app pointing to this repo/zip contents.
2) Set **Secrets** with your OpenAI key:

```
OPENAI_API_KEY = "sk-..."
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run competitorlens.py
```

## Notes
- Basic robots.txt respect.
- For best results, refine prompts (region, B2B vs retail, certifications).
