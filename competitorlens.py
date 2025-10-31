# competitorlens.py (v6) — Minimal UI: single search field, Google-first search
import re, io, json, math, urllib.parse
from typing import List, Optional, Dict
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import tldextract

st.set_page_config(page_title="CompetitorLens — Minimal", page_icon="🔎", layout="wide")

# Optional: only needed if you toggle LLM refine (disabled by default here)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    oai = None

USER_AGENT = "CompetitorLens/0.6-min"
TIMEOUT = 20
SEARCH_RESULTS = 16
MAX_PAGES_PER_DOMAIN = 5
MAX_TOTAL_PAGES = 40
M2_TO_SQFT = 10.7639

# Search keys
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", "")
GOOGLE_CSE_API_KEY = st.secrets.get("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_CX = st.secrets.get("GOOGLE_CSE_CX", "")
BRAVE_API_KEY = st.secrets.get("BRAVE_API_KEY", "")

def domain(url: str) -> str:
    x = tldextract.extract(url); return ".".join([x.domain, x.suffix])

def fetch(u: str):
    try:
        return requests.get(u, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except Exception:
        return None

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script","style","noscript"]): t.decompose()
    import re as _re
    return _re.sub(r"\s+"," ", soup.get_text(" ", strip=True))

# ---- Intent parsing (size + loose location) ----
def parse_intent(prompt: str) -> Dict:
    # size in m² or sq ft, falls back to 10 m² if not found
    M2_TO_SQFT = 10.7639
    area_m2 = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m2|m²|sqm|square\s*meters?)", prompt, re.I)
    if m:
        area_m2 = float(m.group(1))
    else:
        ft = re.search(r"(\d+(?:\.\d+)?)\s*(sq\s*ft|ft2|square\s*feet?)", prompt, re.I)
        if ft:
            area_m2 = float(ft.group(1)) / M2_TO_SQFT
    if not area_m2:
        # common 10x10 ft unit (approx 9.3 m²)
        rect = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:ft|')", prompt, re.I)
        if rect:
            sqft = float(rect.group(1)) * float(rect.group(2))
            area_m2 = sqft / M2_TO_SQFT
    area_m2 = area_m2 or 10.0

    # business type detection
    lower = prompt.lower()
    business = "self storage"
    for k in ["mini warehouse","mini-warehouse","mini storage","mini storage units","mini self storage","storage unit","self storage"]:
        if k in lower:
            business = "mini warehouse" if "warehouse" in k or "mini" in k else "self storage"
            break

    # loose location text (used only to build search queries)
    loc = None
    mloc = re.search(r"(?:in|at|near)\s+([A-Za-z ,.&'-]+)", prompt, re.I)
    if mloc:
        raw = mloc.group(1)
        raw = re.split(r"\b(for|by|with|on)\b", raw)[0].strip(" ,.")
        loc = raw

    return {"business_type": business, "location_hint": loc, "area_m2": area_m2}

# ---- Google-first search ----
def search_google_serpapi(q: str, n: int=10) -> List[str]:
    if not SERPAPI_API_KEY: return []
    try:
        r = requests.get("https://serpapi.com/search.json",
                         params={"engine":"google","q":q,"num":n,"api_key":SERPAPI_API_KEY},
                         headers={"User-Agent":USER_AGENT}, timeout=TIMEOUT)
        if r.status_code!=200: return []
        js = r.json()
        return [it.get("link") for it in js.get("organic_results",[]) if it.get("link","").startswith("http")]
    except Exception:
        return []

def search_google_cse(q: str, n: int=10) -> List[str]:
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX: return []
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1",
                         params={"key": GOOGLE_CSE_API_KEY, "cx": GOOGLE_CSE_CX, "q": q, "num": min(n,10)},
                         headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if r.status_code!=200: return []
        js = r.json()
        return [it.get("link") for it in js.get("items",[]) if it.get("link","").startswith("http")]
    except Exception:
        return []

def search_ddgs(q: str, n: int=10) -> List[str]:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=n, safesearch="moderate", timelimit="y"):
                u = r.get("href") or r.get("url")
                if u and u.startswith("http"): out.append(u)
    except Exception:
        return []
    return out

def search_brave(q: str, n: int=10) -> List[str]:
    if not BRAVE_API_KEY: return []
    try:
        r = requests.get("https://api.search.brave.com/res/v1/web/search",
                         params={"q":q,"count":n},
                         headers={"X-Subscription-Token":BRAVE_API_KEY,"User-Agent":USER_AGENT}, timeout=TIMEOUT)
        if r.status_code!=200: return []
        js = r.json()
        return [it.get("url") for it in js.get("web",{}).get("results",[]) if it.get("url","").startswith("http")]
    except Exception:
        return []

def build_queries(biz: str, loc_hint: Optional[str], area_m2: float) -> List[str]:
    sqft = int(round(area_m2 * M2_TO_SQFT))
    bases = [
        f"{biz} pricing {sqft} sq ft",
        f"{biz} {sqft} sq ft price",
        f"{biz} 10 m² price",
        f"{biz} unit sizes prices",
        f"{biz} rates monthly",
        f"{biz} '10x10' price",
        f"{biz} storage rates pdf",
    ]
    if loc_hint:
        bases = [f"{b} {loc_hint}" for b in bases]
    seen=set(); out=[]
    for q in bases:
        q2=" ".join(q.split())
        if q2 not in seen:
            out.append(q2); seen.add(q2)
    return out

def search_all(qs: List[str], limit:int=SEARCH_RESULTS) -> List[str]:
    urls = []
    for q in qs:
        urls += [u for u in search_google_serpapi(q, n=limit//2) if u not in urls]
        if len(urls) < limit:
            urls += [u for u in search_google_cse(q, n=limit//2) if u not in urls]
        if len(urls) < limit:
            urls += [u for u in search_ddgs(q, n=limit//2) if u not in urls]
        if len(urls) < limit:
            urls += [u for u in search_brave(q, n=limit//2) if u not in urls]
        if len(urls) >= limit:
            break
    # Deduplicate by domain
    seen, out = set(), []
    for u in urls:
        d=domain(u)
        if d in seen: continue
        seen.add(d); out.append(u)
        if len(out)>=limit: break
    return out

# ---- Extraction ----
PRICE_RE = re.compile(r"(?<!\w)(?:USD|\$)\s?(\d{2,5}(?:[.,]\d{2})?)\s*(?:/|\s*(?:per|a)\s*)?(month|mo|wk|week|day|year|yr)?", re.I)
RATE_SQFT_RE = re.compile(r"\$?\s?(\d+(?:\.\d+)?)\s*/\s*(?:sq\s*ft|ft²)\s*/?\s*(?:mo|month)?", re.I)
SIZE_M2_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|square\s*meters?)", re.I)
SIZE_SQFT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:sq\s*ft|ft2|ft²|square\s*feet?)", re.I)
SIZE_RECT_FT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:ft|')", re.I)

def period_to_month(amount: float, period: Optional[str]) -> float:
    if not period: return amount
    p=period.lower()
    if p in ["mo","month","monthly"]: return amount
    if p in ["wk","week"]: return amount*4.345
    if p=="day": return amount*30.4375
    if p in ["year","yr","annual"]: return amount/12.0
    return amount

def nearest_area(text: str, target_m2: float):
    m2s=[float(m.group(1)) for m in SIZE_M2_RE.finditer(text)]
    if m2s:
        v=min(m2s, key=lambda x: abs(x-target_m2)); return v, f"{v:.1f} m²"
    fts=[float(m.group(1)) for m in SIZE_SQFT_RE.finditer(text)]
    if fts:
        v=min(fts, key=lambda x: abs(x/ M2_TO_SQFT - target_m2)); return v/M2_TO_SQFT, f"{int(round(v))} sq ft"
    rect=[float(a.group(1))*float(a.group(2)) for a in SIZE_RECT_FT_RE.finditer(text)]
    if rect:
        v=min(rect, key=lambda x: abs(x/ M2_TO_SQFT - target_m2)); return v/M2_TO_SQFT, f"{int(round(v))} sq ft (rect)"
    return None, ""

def extract_offers(site_domain: str, page_url: str, text: str, target_m2: float):
    offers=[]
    for m in RATE_SQFT_RE.finditer(text):
        rate=float(m.group(1))
        per_m2=(rate*(target_m2*M2_TO_SQFT))/target_m2
        offers.append([site_domain, page_url, "rate per sq ft", rate, "$", "per sq ft per month", per_m2, "derived from per-sqft rate"])
    for m in PRICE_RE.finditer(text):
        amt=float(m.group(1).replace(",",""))
        per=m.group(2) or "month"
        monthly=period_to_month(amt, per)
        a,label=nearest_area(text, target_m2)
        ppm2 = monthly/a if a else float("nan")
        offers.append([site_domain, page_url, label or "N/A", monthly, "$", "month", ppm2, "matched nearest listed size" if a else "price found but size unspecified"])
    return offers

def crawl_extract(urls: List[str], target_m2: float, per_domain:int=MAX_PAGES_PER_DOMAIN):
    from collections import deque
    results=[]; seen=set(); counts={}
    for root in urls:
        d=domain(root)
        if counts.get(d,0)>=per_domain: continue
        q=deque([root]); pages=0
        while q and pages<per_domain and len(seen)<MAX_TOTAL_PAGES:
            u=q.popleft()
            if u in seen: continue
            seen.add(u)
            r=fetch(u)
            if not r or r.status_code>=400: continue
            txt=clean_text(r.text)
            if any(k in txt.lower() for k in ["pricing","rates","unit sizes","size guide","storage units","rent now","reserve","pricing &"]):
                results += extract_offers(d, u, txt, target_m2)
            soup=BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                nu=urllib.parse.urljoin(u, a["href"])
                if domain(nu)!=d or nu in seen: continue
                if any(p in nu.lower() for p in ["/pricing","/rates","/sizes","/size-guide","/units","/storage","/rent","/reserve","/locations"]):
                    q.append(nu)
            pages+=1
        counts[d]=counts.get(d,0)+pages
    return results

def refine_with_llm(rows, target_m2: float):
    if not oai or not rows: return rows
    payload={"target_m2":target_m2,"offers":[{
        "business": r[0],"website": r[1],"advertised_size": r[2],"price_amount": r[3],
        "currency": r[4],"period": r[5],"price_per_m2_month": r[6],"note": r[7]} for r in rows]}
    try:
        resp=oai.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, response_format={"type":"json_object"},
            messages=[{"role":"system","content":"Select one representative offer per business close to target size. Return JSON {'selected': [...]}"},
                      {"role":"user","content":json.dumps(payload)}])
        data=json.loads(resp.choices[0].message.content)
        sel=[]
        for d in data.get("selected", []):
            sel.append([d.get("business",""), d.get("website",""), d.get("advertised_size",""),
                        float(d.get("price_amount",0.0)), d.get("currency","$"), d.get("period","month"),
                        float(d.get("price_per_m2_month",float('nan'))), d.get("note","")])
        return sel or rows
    except Exception:
        return rows

# ---- UI (single field) ----
st.title("🔎 CompetitorLens — Minimal")
st.caption("Type a research request. Example: “Find the pricing of mini warehouses in New York for 10m2 warehouses”.")

prompt = st.text_input("Your request", value="Find the pricing of mini warehouses in New York for 10m2 warehouses")
use_llm = st.checkbox("Use LLM to refine noisy offers (optional)", value=False)
run = st.button("Search & Analyze", type="primary")

if run:
    intent = parse_intent(prompt)
    biz, loc_hint, target = intent["business_type"], intent["location_hint"], intent["area_m2"]
    st.markdown("**Parsed intent**")
    st.json(intent)

    st.subheader("1) Search")
    qs = build_queries(biz, loc_hint, target)
    st.write(qs)
    urls = search_all(qs, limit=SEARCH_RESULTS)
    if not urls:
        st.error("No search hits. Try specifying a city or size (e.g., '10x10' or '100 sq ft').")
        st.stop()

    with st.expander("Websites to crawl"):
        for u in urls:
            st.write("- ", u)

    st.subheader("2) Crawl & extract prices")
    rows = crawl_extract(urls, target_m2=target)
    if not rows:
        st.warning("No public prices found. Many providers gate pricing behind ZIP/date. Try different wording (e.g., 'rates', 'unit sizes', '10x10').")
        st.stop()

    if use_llm:
        rows = refine_with_llm(rows, target)

    df = pd.DataFrame(rows, columns=["Business (domain)","Website","Advertised size","Price amount (monthly USD)","Currency","Period","Price per m² / month (USD)","Notes"])
    st.dataframe(df, use_container_width=True)

    ser=pd.to_numeric(df["Price per m² / month (USD)"], errors="coerce").dropna()
    if not ser.empty:
        q1,q3=ser.quantile(0.25), ser.quantile(0.75); iqr=q3-q1
        trimmed=ser[(ser>=q1-1.5*iqr)&(ser<=q3+1.5*iqr)]
        avg=trimmed.mean()
        st.markdown(f"### ✅ Estimated average for ~{target:.1f} m² (~{target*M2_TO_SQFT:.0f} sq ft): **${avg:.2f} per m² / month**")
        st.markdown(f"That’s roughly **${avg/ M2_TO_SQFT:.2f} per sq ft / month**.")
    else:
        st.info("No normalized prices to compute average.")

    st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), "prices.csv", mime="text/csv")
    md = df.to_markdown(index=False)
    st.download_button("📝 Download Markdown", md.encode("utf-8"), "prices.md", mime="text/markdown")
