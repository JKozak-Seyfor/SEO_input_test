import os
import io
import re
import json
import time
import typing as t
from dataclasses import dataclass

import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI import (použije se jen v API režimu; app funguje i bez něj)
try:
    import openai
except Exception:
    openai = None


# -------------------------------
# Config & helpers
# -------------------------------

@dataclass
class Limits:
    title_min: int = 30
    title_max: int = 50
    h1_min: int = 20
    h1_max: int = 40
    desc_min: int = 140
    desc_max: int = 160
    body_min: int = 1200

DEFAULT_LIMITS = Limits()

BANNED_PATTERN_TITLE = re.compile(r"[!]", flags=0)
URL_PATTERN = re.compile(r"https?://|www\.", re.I)


def normalize_keyword(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = s.strip("/")
    s = s.replace("-", " ")
    s = re.sub(r"[_/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.split()[0] if s else ""


def ensure_len(text: str, min_len: int | None = None, max_len: int | None = None) -> str:
    t = (text or "").strip()
    if min_len and len(t) < min_len:
        t += " " + (" ".join(["…" for _ in range(min_len - len(t))]))
    if max_len and len(t) > max_len:
        t = t[:max_len]
        t = re.sub(r"\s+\S*$", "", t)
    return t


def simple_stub_generate(keyword: str, limits: Limits) -> dict:
    kw = keyword.capitalize() or "Téma"
    page_title = ensure_len(f"{kw} – průvodce a tipy", limits.title_min, limits.title_max)
    h1 = ensure_len(f"{kw} – novinky a přehled", limits.h1_min, limits.h1_max)
    desc = ensure_len(
        f"{kw} v kostce: co to je, k čemu slouží a jak z něj vytěžit maximum. "
        f"Praktické rady a stručné tipy v jednom místě.",
        limits.desc_min,
        limits.desc_max,
    )
    body = (
        f"<h3>{kw}</h3>\n{kw} patří mezi témata, která lidé často hledají, "
        f"ale zároveň u nich narážejí na rozporné informace. V tomto přehledu shrnujeme klíčové pojmy, přínosy a omyly."
    )
    if len(body) < limits.body_min:
        body += " " + " ".join(["Rozšiřující text." for _ in range(80)])

    page_title_en = ensure_len(f"{kw} – guide and tips", limits.title_min, limits.title_max)
    h1_en = ensure_len(f"{kw} – updates and overview", limits.h1_min, limits.h1_max)
    desc_en = ensure_len(
        f"{kw} explained: what it is, where it helps and how to make the most of it. Practical advice and concise tips in one place.",
        limits.desc_min,
        limits.desc_max,
    )
    return {
        "page_title": page_title,
        "title_for_newest_advertisement_list": h1,
        "description": desc,
        "text_on_page": ensure_len(body, min_len=limits.body_min),
        "page_title_eng": page_title_en,
        "title_for_newest_advertisement_list_eng": h1_en,
        "description_eng": desc_en,
        "text_on_page_eng": ensure_len(body, min_len=limits.body_min),
    }


def similar_title(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def similar_long(a: str, b: str) -> float:
    vect = TfidfVectorizer(ngram_range=(3, 5), min_df=1)
    X = vect.fit_transform([a or "", b or ""])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def validate_row(rec: dict, limits: Limits) -> list[str]:
    errs = []
    if not (limits.title_min <= len(rec["page_title"]) <= limits.title_max):
        errs.append("page_title length out of range")
    if not (limits.h1_min <= len(rec["title_for_newest_advertisement_list"]) <= limits.h1_max):
        errs.append("H1 length out of range")
    if not (limits.desc_min <= len(rec["description"]) <= limits.desc_max):
        errs.append("description length out of range")
    if not (len(rec["text_on_page"]) >= limits.body_min):
        errs.append("text_on_page too short")
    if BANNED_PATTERN_TITLE.search(rec["page_title"]):
        errs.append("page_title contains '!'")
    if URL_PATTERN.search(rec["description"]):
        errs.append("description contains URL")
    return errs


# -------------------------------
# OpenAI batch generator (GPT-5 nano)
# -------------------------------

def generate_batch_openai(api_key: str, keywords: list[str], limits: Limits, hard_negatives: list[str]) -> list[dict]:
    if openai is None or not api_key:
        return [simple_stub_generate(k, limits) for k in keywords]

    openai.api_key = api_key
    limits_payload = {
        "page_title": {"min": limits.title_min, "max": limits.title_max},
        "h1": {"min": limits.h1_min, "max": limits.h1_max},
        "description": {"min": limits.desc_min, "max": limits.desc_max},
        "text_on_page_min": limits.body_min,
    }

    sys_prompt = (
        "You are a skilled Czech SEO copywriter specializing in tasteful adult-oriented content. "
        "Generate unique, structured website texts in Czech and English exactly according to the JSON format provided. "
        "No exclamation marks, no vulgarities, no URLs. Titles and descriptions must fit the length limits strictly."
    )

    items = [{"idx": i, "name": k} for i, k in enumerate(keywords)]

    user_prompt = f"""
Máš pole objektů ({{idx, name}}):
{json.dumps(items, ensure_ascii=False)}

Pro každou položku vytvoř objekt:
{{
  "idx": <číslo>,
  "page_title": "...",  // 30–50 znaků, CZ, přirozeně obsahuje keyword, benefit/lákadlo
  "description": "...", // 140–160 znaků, CZ, shrnutí + jemné CTA, bez URL
  "text_on_page": "...", // ≥1200 znaků, CZ, HTML-like: úvod → 2–3 H3 podtémata → závěr
  "title_for_newest_advertisement_list": "...", // 20–40 znaků, CZ H1, krátké a úderné
  "page_title_eng": "...", // EN varianta, stejné délky
  "description_eng": "...", // EN varianta, stejné délky
  "text_on_page_eng": "...", // EN text, strukturovaný
  "title_for_newest_advertisement_list_eng": "..." // EN H1
}}

Pravidla:
1) Dodrž délky: Title {limits.title_min}-{limits.title_max}, H1 {limits.h1_min}-{limits.h1_max}, Desc {limits.desc_min}-{limits.desc_max}, text ≥{limits.body_min}.
2) Bez vykřičníků, emoji a URL.
3) Klíčové slovo použij v Title, H1, Description a úvodu textu.
4) Styl: smyslný, ale decentní; důraz na komfort, diskrétnost, zážitek.
5) Vyhni se podobnostem s těmito titulky: {json.dumps(hard_negatives[:20], ensure_ascii=False)}
6) Výsledek vrať jako čisté JSON pole, žádný doprovodný text.

Délkové limity:
{json.dumps(limits_payload, ensure_ascii=False, indent=2)}
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-5-nano",  # hlavní model
            fallback_models=["gpt-4o-mini"],  # záloha
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=4000,
            frequency_penalty=0.2,
        )
        content = resp["choices"][0]["message"]["content"]
        start, end = content.find("["), content.rfind("]")
        arr = json.loads(content[start:end+1])
        out_by_idx = {int(x["idx"]): x for x in arr}

        results = []
        for i, k in enumerate(keywords):
            x = out_by_idx.get(i)
            results.append(simple_stub_generate(k, limits) if not x else {
                "page_title": x["page_title"],
                "title_for_newest_advertisement_list": x["title_for_newest_advertisement_list"],
                "description": x["description"],
                "text_on_page": x["text_on_page"],
                "page_title_eng": x["page_title_eng"],
                "title_for_newest_advertisement_list_eng": x["title_for_newest_advertisement_list_eng"],
                "description_eng": x["description_eng"],
                "text_on_page_eng": x["text_on_page_eng"],
            })
        return results
    except Exception:
        return [simple_stub_generate(k, limits) for k in keywords]


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="CSV Content Generator", layout="wide")
st.title("CSV Content Generator (CZ/EN) — GPT-5 nano")

with st.sidebar:
    st.header("Režim generování")
    mode = st.selectbox("Zvol režim", ["Template (offline stub)", "OpenAI API"], index=1)
    limits = Limits()
    batch_size = st.number_input("Batch size", 5, 100, 20)
    chunk_sleep = st.slider("Prodleva mezi dávkami (s)", 0.0, 5.0, 0.0, 0.5)
    api_key = st.text_input("OpenAI API key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))

uploaded = st.file_uploader("Nahraj CSV nebo XLSX", type=["csv", "xlsx"])
if uploaded is not None and isinstance(uploaded, list):
    uploaded = uploaded[0]

if uploaded:
    if hasattr(uploaded, "name") and uploaded.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        data = uploaded.getvalue().decode("utf-8", errors="ignore")
        try:
            df = pd.read_csv(io.StringIO(data), sep=";")
        except Exception:
            df = pd.read_csv(io.StringIO(data), sep=",")

    st.subheader("Náhled")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    name_col = st.selectbox("Sloupec s Name", cols, index=cols.index("name") if "name" in cols else 0)

    if st.button("Generovat dávkově"):
        out_rows, logs = [], []
        prev_titles, prev_bodies = [], []

        progress_text = st.empty()
        progress_bar = st.progress(0)
        rows = list(df.itertuples(index=True))
        total = len(rows)
        done = 0

        all_keywords = [normalize_keyword(str(getattr(r, name_col))) for r in rows]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_keywords = all_keywords[start:end]
            hard_neg = list(dict.fromkeys(prev_titles))[:30]

            batch_out = generate_batch_openai(api_key, batch_keywords, limits, hard_neg)

            for i, rec in enumerate(batch_out):
                idx_in_df = rows[start + i].Index
                row = df.iloc[idx_in_df]
                keyword = batch_keywords[i]
                errs = validate_row(rec, limits)

                prev_titles.append(rec["page_title"])
                prev_bodies.append(rec["text_on_page"])

                out_rows.append({
                    "name": row.get(name_col, ""),
                    "page_title": rec["page_title"],
                    "page_title_eng": rec["page_title_eng"],
                    "description": rec["description"],
                    "description_eng": rec["description_eng"],
                    "text_on_page": rec["text_on_page"],
                    "text_on_page_eng": rec["text_on_page_eng"],
                    "title_for_newest_advertisement_list": rec["title_for_newest_advertisement_list"],
                    "title_for_newest_advertisement_list_eng": rec["title_for_newest_advertisement_list_eng"],
                })
                logs.append({"row": int(idx_in_df), "keyword": keyword, "errors": errs})

            done = end
            progress_bar.progress(int(done / total * 100))
            progress_text.text(f"Zpracováno {done}/{total} (dávka {start+1}–{end})")
            if chunk_sleep > 0:
                time.sleep(chunk_sleep)

        progress_bar.empty()
        progress_text.text("✅ Hotovo – generování dávkami dokončeno.")
        out_df = pd.DataFrame(out_rows)
        st.dataframe(out_df.head())
        st.download_button("Stáhnout CSV", out_df.to_csv(index=False).encode("utf-8"), "doplnene_texty.csv")
        st.json(logs)
else:
    st.info("Nahraj CSV/XLSX se sloupcem 'name'.")
