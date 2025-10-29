# competitorlens.py
# ------------------------------------------------------------
# CompetitorLens — prompt-driven competitor/product research
# ------------------------------------------------------------
# pip install -r requirements.txt
# streamlit run competitorlens.py
# ------------------------------------------------------------

import re
import io
import json
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

import requests
from bs4 import BeautifulSoup
import tldextract
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# ------------- Config -------------
st.set_page_config(page_title="CompetitorLens", page_icon="🕵️", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# OpenAI client (optional fallback if package missing)
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    oai = None

USER_AGENT = "CompetitorLensBot/0.1 (+research; respectful)"
TIMEOUT = 15
MAX_TOTAL_PAGES = 30  # global crawl cap
MAX_PER_DOMAIN = 6    # per-site cap
SEARCH_RESULTS = 10   # initial search breadth
CHUNK_CHAR_LIMIT = 6000  # per-site text passed to LLM
PDF_BYTE_LIMIT = 1_5 * 1024 * 1024  # ~1.5MB cap

# ------------- Utilities -------------

def normalize_url(u: str) -> str:
    try:
        return urllib.parse.urlsplit(u)._replace(fragment="").geturl()
    except Exception:
        return u

def same_domain(u1: str, u2: str) -> bool:
    a = tldextract.extract(u1)
    b = tldextract.extract(u2)
    return (a.domain, a.suffix) == (b.domain, b.suffix)

def fetch(url: str) -> Optional[requests.Response]:
    try:
        return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except Exception:
        return None

def read_robots_allow(domain_root: str) -> bool:
    robots = urllib.parse.urljoin(domain_root, "/robots.txt")
    try:
        r = requests.get(robots, headers={"User-Agent": USER_AGENT}, timeout=5)
        if r.status_code != 200:
            return True  # assume allowed if no robots
        txt = r.text.lower()
        if "user-agent: *" in txt and "disallow: /" in txt:
            return False
        return True
    except Exception:
        return True

def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    texts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "th", "td"]):
        t = el.get_text(" ", strip=True)
        if t:
            texts.append(t)
    joined = " ".join(texts)
    return re.sub(r"\s+", " ", joined).strip()

def looks_like_pdf(resp: requests.Response, url: str) -> bool:
    ct = resp.headers.get("Content-Type", "").lower()
    return "application/pdf" in ct or url.lower().endswith(".pdf")

def parse_pdf_text(resp: requests.Response) -> str:
    try:
        if len(resp.content) > PDF_BYTE_LIMIT:
            return ""
        from pdfminer.high_level import extract_text
        with io.BytesIO(resp.content) as f:
            return extract_text(f)[:15000]
    except Exception:
        return ""

def is_producty(text: str) -> bool:
    keys = ["price", "€", "$", "£", "catalog", "range", "spec", "capacity",
            "sizes", "SKU", "certification", "DOP", "PDO", "PGI", "organic",
            "wholesale", "export", "product", "bottle", "litre", "500ml", "750ml"]
    score = sum(1 for k in keys if k.lower() in text.lower())
    return score >= 2

# ------------- Search -------------

def ddg_search(query: str, count: int = SEARCH_RESULTS) -> List[str]:
    q = urllib.parse.urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{q}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if not href:
                continue
            if "uddg=" in href:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                href = qs.get("uddg", [href])[0]
            links.append(normalize_url(href))
            if len(links) >= count:
                break
        return links
    except Exception:
        return []

def dedupe_domains(urls: List[str], max_per_domain: int = 2) -> List[str]:
    seen = {}
    out = []
    for u in urls:
        x = tldextract.extract(u)
        key = (x.domain, x.suffix)
        seen.setdefault(key, 0)
        if seen[key] < max_per_domain:
            out.append(u)
            seen[key] += 1
    return out

# ------------- Crawl -------------

from dataclasses import dataclass

@dataclass
class PageGrab:
    url: str
    text: str

@dataclass
class CrawlResult:
    root: str
    pages: List[PageGrab] = field(default_factory=list)

def crawl_site(root_url: str, limit: int = MAX_PER_DOMAIN) -> CrawlResult:
    root_url = normalize_url(root_url)
    base = urllib.parse.urljoin(root_url, "/")
    if not read_robots_allow(base):
        return CrawlResult(root=root_url, pages=[])

    seen: Set[str] = set()
    q = deque([root_url])
    grabbed: List[PageGrab] = []

    while q and len(grabbed) < limit:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        resp = fetch(u)
        if not resp or resp.status_code >= 400:
            continue

        text = ""
        if looks_like_pdf(resp, u):
            text = parse_pdf_text(resp)
        else:
            text = extract_visible_text(resp.text)

        if text and is_producty(text):
            grabbed.append(PageGrab(url=u, text=text))

        if len(grabbed) < limit and resp and not looks_like_pdf(resp, u):
            try:
                soup = BeautifulSoup(resp.text, "lxml")
                for a in soup.find_all("a", href=True):
                    nu = urllib.parse.urljoin(u, a["href"])
                    nu = normalize_url(nu)
                    if tldextract.extract(nu).domain == tldextract.extract(root_url).domain and nu not in seen:
                        if any(part in nu.lower() for part in ["/product", "/shop", "/catalog", "/range", "/olive", "/oil"]):
                            q.append(nu)
            except Exception:
                pass

    return CrawlResult(root=root_url, pages=grabbed)

# ------------- LLM Extraction -------------

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "comparison": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "website": {"type": "string"},
                    "product_range": {"type": "string"},
                    "certifications": {"type": "string"},
                    "avg_price": {"type": "string"},
                    "price_basis": {"type": "string"},
                    "unique_selling_points": {"type": "string"},
                    "notable_skus": {"type": "string"}
                },
                "required": ["company", "website", "product_range", "avg_price", "unique_selling_points"]
            }
        },
        "executive_summary": {"type": "string"}
    },
    "required": ["comparison", "executive_summary"]
}

SYS_PROMPT = """You are a rigorous market research analyst.
You will receive raw text from multiple websites (exporters, brands, wholesalers).
TASK:
1) Identify the company and website for each source.
2) Extract product ranges, certifications (e.g., Organic, DOP/PDO/PGI, IFS/BRC), prices or price ranges (note if absent),
   price basis (per litre, per 500ml, wholesale/retail), notable SKUs/sizes, and unique selling points.
3) Return STRICT JSON matching the provided schema. Do not include extra keys or markdown.
4) Be conservative: if price is missing, set "avg_price": "N/A" and note in "price_basis".
5) Merge near-duplicate companies (fuzzy match of names > 90)."""

def call_llm_extract(sources: List[Tuple[str, str]], user_focus: str) -> Dict:
    if not oai:
        raise RuntimeError("OpenAI API key missing. Add it to .streamlit/secrets.toml as OPENAI_API_KEY.")
    bundles = []
    for url, text in sources:
        trimmed = text[:CHUNK_CHAR_LIMIT]
        bundles.append(f"=== SOURCE BEGIN ===\nURL: {url}\nTEXT: {trimmed}\n=== SOURCE END ===")
    payload = "\n\n".join(bundles)

    user_msg = (
        f"Research focus: {user_focus}\n"
        f"Please extract structured data.\n\n"
        f"{payload}"
    )

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYS_PROMPT + "\nJSON Schema:\n" + json.dumps(JSON_SCHEMA)},
            {"role": "user", "content": user_msg},
        ],
    )
    content = resp.choices[0].message.content
    return json.loads(content)

def merge_near_duplicates(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        name = r.get("company", "").strip()
        matched = None
        for i, o in enumerate(out):
            if fuzz.token_sort_ratio(name, o.get("company", "")) >= 90:
                matched = i
                break
        if matched is None:
            out.append(r)
        else:
            for k, v in r.items():
                if not v or v == "N/A":
                    continue
                if not out[matched].get(k) or out[matched][k] == "N/A":
                    out[matched][k] = v
    return out

def json_to_df(payload: Dict):
    rows = payload.get("comparison", [])
    rows = merge_near_duplicates(rows)
    df = pd.DataFrame(rows)
    preferred_cols = ["company","website","product_range","certifications","avg_price","price_basis","notable_skus","unique_selling_points"]
    for c in preferred_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[preferred_cols]
    summary = payload.get("executive_summary", "")
    return df, summary

# ------------- UI -------------

st.title("🕵️ CompetitorLens — Prompt-Driven Product & Competitor Research")
st.caption("Enter a research prompt. I’ll search, crawl, extract, and compare — then give you a table + executive summary.")

colA, colB = st.columns([3,1])
with colA:
    prompt = st.text_input(
        "Your research prompt",
        value="Compare olive oil exporters in France by product range, certifications, and price",
        help="Be specific about region, product type, and comparison axes."
    )
with colB:
    top_sites = st.number_input("Max sites to analyze", min_value=3, max_value=20, value=8, step=1)
depth = st.slider("Max pages per domain", 2, 10, 5, help="Deeper crawl may take longer.")
agree = st.checkbox("I’ll use this responsibly and respect robots.txt", value=True)

run = st.button("Run analysis", type="primary")

if run:
    if not OPENAI_API_KEY:
        st.error("Add your OpenAI API key in .streamlit/secrets.toml (OPENAI_API_KEY).")
        st.stop()
    if not agree:
        st.warning("Please confirm responsible use first.")
        st.stop()

    # --- Search
    st.subheader("1) 🔎 Web search")
    st.write("Searching for relevant companies and product/catalog pages...")

    def ddg_search_local(query, count=SEARCH_RESULTS):
        return ddg_search(query, count=count)

    urls = ddg_search_local(prompt, count=SEARCH_RESULTS*2)
    urls = [u for u in urls if u.startswith("http")]
    urls = dedupe_domains(urls, max_per_domain=1)[:SEARCH_RESULTS]
    if not urls:
        st.error("No search results found. Try rephrasing the prompt.")
        st.stop()

    with st.expander("Search hits"):
        for u in urls:
            st.write(f"- {u}")

    # --- Crawl
    st.subheader("2) 🕷️ Crawling sites")
    total_pages = 0
    sources: List[Tuple[str,str]] = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, root in enumerate(urls[:top_sites], start=1):
        status.write(f"Crawling {root} ({i}/{min(top_sites, len(urls))}) …")
        cr = crawl_site(root, limit=min(depth, MAX_PER_DOMAIN))
        for pg in cr.pages:
            total_pages += 1
            sources.append((pg.url, pg.text))
        progress.progress(min(1.0, i / max(1, min(top_sites, len(urls)))))
        if total_pages >= MAX_TOTAL_PAGES:
            break

    if not sources:
        st.warning("Crawl finished but no product-like pages were detected. Try increasing depth or broadening the prompt.")
        st.stop()

    with st.expander("Crawled pages used for analysis"):
        for url, _ in sources:
            st.write(f"- {url}")

    # --- LLM Extraction
    st.subheader("3) 🤖 LLM extraction & comparison")
    with st.spinner("Asking the model for structured fields and a comparison…"):
        try:
            payload = call_llm_extract(sources, prompt)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    # --- Table + Summary
    st.subheader("4) 📊 Results")
    df, summary = json_to_df(payload)

    if df.empty:
        st.warning("No rows produced. The model didn’t find comparable companies.")
    else:
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "comparison.csv", mime="text/csv")
        with c2:
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Comparison")
            st.download_button("📊 Download Excel", xls_buf.getvalue(), "comparison.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            md = df.to_markdown(index=False)
            md_full = f"## Comparison\n\n{md}\n\n## Executive Summary\n\n{summary}"
            st.download_button("📝 Download Markdown", md_full.encode("utf-8"), "analysis.md", mime="text/markdown")

        st.markdown("### 🧾 Executive Summary")
        st.write(summary)

    st.caption("Tip: refine your prompt to include niches (e.g., 'B2B exporters', 'wholesale', 'PDO/DOP certified'). You can also bump crawl depth for catalogs.")
