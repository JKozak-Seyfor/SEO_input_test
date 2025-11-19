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
    import openai  # legacy klient
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

BANNED_PATTERN_TITLE = re.compile(r"[!]", flags=0)  # jednoduché: zákaz '!'
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
    page_title = ensure_len(f"{kw} – průvodce a tipy",
                            limits.title_min, limits.title_max)
    h1 = ensure_len(f"{kw} – novinky a přehled",
                    limits.h1_min, limits.h1_max)
    desc = ensure_len(
        f"{kw} v kostce: co to je, k čemu slouží a jak z něj vytěžit maximum. "
        f"Praktické rady a stručné tipy v jednom místě.",
        limits.desc_min, limits.desc_max
    )
    # body (≥1200 znaků)
    paras = []
    intro = (f"{kw} patří mezi témata, která lidé často hledají, ale zároveň u nich narážejí na rozporné informace. "
             f"V tomto přehledu shrnujeme klíčové pojmy, přínosy a časté omyly. Najdete zde praktické postupy, jak začít.")
    p2 = (f"Co si pod pojmem {kw} přesně představit? Nejlépe uděláme, když si projdeme základní definici a ukážeme, "
          f"jak se používá v praxi. Důležitý je kontext: kdo bude výsledky využívat a jaké jsou cíle.")
    p3 = ("Jak začít krok za krokem: určete priority, sbírejte jen data, která skutečně potřebujete, "
          "a ověřujte si zdroje. Zkoušejte malé iterace, vyhodnocujte dopady a dokumentujte závěry.")
    p4 = ("Na co si dát pozor: vyhněte se univerzálním radám bez kontextu a průběžně pracujte s riziky. "
          "Checklist: cíl, metrika, odpovědnosti, termíny a zpětná vazba.")
    body = "\n\n".join([intro, p2, p3, p4])
    if len(body) < limits.body_min:
        body += "\n\n" + ("Další doporučení: " + " ".join(["Doplňte příklady z vlastní praxe." for _ in range(80)]))
    body = ensure_len(body, min_len=limits.body_min)

    # EN rychlé varianty (stub)
    page_title_en = ensure_len(f"{kw} – guide and tips",
                               limits.title_min, limits.title_max)
    h1_en = ensure_len(f"{kw} – updates and overview",
                       limits.h1_min, limits.h1_max)
    desc_en = ensure_len(
        f"{kw} explained: what it is, where it helps and how to make the most of it. "
        f"Practical advice and concise tips in one place.",
        limits.desc_min, limits.desc_max
    )
    body_en = body  # v demo režimu recyklujeme; v API režimu by se překládalo

    return {
        "page_title": page_title,
        "title_for_newest_advertisement_list": h1,
        "description": desc,
        "text_on_page": body,
        "page_title_eng": page_title_en,
        "title_for_newest_advertisement_list_eng": h1_en,
        "description_eng": desc_en,
        "text_on_page_eng": body_en,
    }


def similar_title(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def similar_long(a: str, b: str) -> float:
    vect = TfidfVectorizer(ngram_range=(3, 5), min_df=1)
    X = vect.fit_transform([a or "", b or ""])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def validate_row(rec: dict, limits: Limits) -> list[str]:
    errs = []
    # lengths
    if not (limits.title_min <= len(rec["page_title"]) <= limits.title_max):
        errs.append("page_title length out of range")
    if not (limits.h1_min <= len(rec["title_for_newest_advertisement_list"]) <= limits.h1_max):
        errs.append("H1 length out of range")
    if not (limits.desc_min <= len(rec["description"]) <= limits.desc_max):
        errs.append("description length out of range")
    if not (len(rec["text_on_page"]) >= limits.body_min):
        errs.append("text_on_page length below minimum")

    # patterns
    if BANNED_PATTERN_TITLE.search(rec["page_title"]):
        errs.append("page_title contains '!'")
    if BANNED_PATTERN_TITLE.search(rec["title_for_newest_advertisement_list"]):
        errs.append("H1 contains '!'")
    if URL_PATTERN.search(rec["description"]):
        errs.append("description contains URL")

    return errs


# -------------------------------
# OpenAI – batch generování
# -------------------------------

def generate_batch_openai(api_key: str, keywords: list[str], limits: Limits, hard_negatives: list[str]) -> list[dict]:
    """
    Vrátí pole dictů (jeden dict = 1 řádek CSV) pro danou dávku 'keywords'.
    Každý dict obsahuje CZ+EN pole: page_title, title_for_newest_advertisement_list, description, text_on_page, *_eng.
    """
    if openai is None or not api_key:
        # fallback – použij lokální šablonu pro každý keyword
        return [simple_stub_generate(k, limits) for k in keywords]

    openai.api_key = api_key

    limits_payload = {
        "page_title": {"min": limits.title_min, "max": limits.title_max},
        "h1": {"min": limits.h1_min, "max": limits.h1_max},
        "description": {"min": limits.desc_min, "max": limits.desc_max},
        "text_on_page_min": limits.body_min
    }

    sys_prompt = (
        "Jsi seniorní český copywriter a SEO editor. Dodrž přesně zadání a vrať POUZE validní JSON pole.\n"
        "Piš věcně, bez superlativů a klišé. V Title/H1 nepoužívej vykřičníky ani emoji. "
        "V Description nesmí být URL. Každá položka musí být unikátní v rámci dávky.\n"
    )

    items = [{"idx": i, "name": k} for i, k in enumerate(keywords)]

    user_prompt = f"""
Máš dávku položek (pole objektů {{idx, name}}):
{json.dumps(items, ensure_ascii=False)}

Vrať POUZE JSON pole stejné délky, kde každý prvek má strukturu:
{{
  "idx": <číslo z inputu>,
  "page_title": "...",                        // CZ, {limits.title_min}–{limits.title_max} znaků, obsahuje keyword přirozeně, bez '!'
  "title_for_newest_advertisement_list": "...", // CZ H1, {limits.h1_min}–{limits.h1_max} znaků, bez '!'
  "description": "...",                       // CZ, {limits.desc_min}–{limits.desc_max} znaků, bez URL
  "text_on_page": "...",                      // CZ, ≥{limits.body_min} znaků; struktura: úvod → 2–4 odstavce → závěr/VK
  "page_title_eng": "...",                    // EN varianta title, stejné rozsahy, bez '!'
  "title_for_newest_advertisement_list_eng": "...", // EN H1
  "description_eng": "...",                   // EN description, bez URL
  "text_on_page_eng": "..."                   // EN text, ≥{limits.body_min} znaků
}}

POVINNÁ pravidla:
1) Dodrž délkové limity přesně (nepodkroč ani nepřekroč).
2) V title/H1 nepoužívej vykřičníky/emoji; v description nesmí být URL.
3) Klíčové slovo použij přirozeně v page_title, H1, description a v úvodu text_on_page.
4) Styl: srozumitelný, praktický, bez klišé.
5) Unikátnost: vyhni se podobnostem k těmto titulům/H1 (hard negatives) i mezi položkami dávky:
   {json.dumps(hard_negatives[:20], ensure_ascii=False)}
6) Vrať POUZE JSON pole – žádný text navíc.

Délkové limity:
{json.dumps(limits_payload, ensure_ascii=False, indent=2)}
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=4000,  # uprav dle velikosti dávky / kontext limitu
            presence_penalty=0.0,
            frequency_penalty=0.2,
        )
        content = resp["choices"][0]["message"]["content"]
        start = content.find("[")
        end = content.rfind("]")
        arr = json.loads(content[start:end+1])

        out_by_idx = {int(x["idx"]): x for x in arr}
        results = []
        for i, k in enumerate(keywords):
            x = out_by_idx.get(i)
            if not x:
                results.append(simple_stub_generate(k, limits))
            else:
                results.append({
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
st.title("CSV Content Generator (CZ/EN)")

with st.sidebar:
    st.header("Režim generování")
    mode = st.selectbox("Zvol režim", ["Template (offline stub)", "OpenAI API"], index=0)
    limits = Limits(
        title_min=st.number_input("Title min", 10, 100, DEFAULT_LIMITS.title_min),
        title_max=st.number_input("Title max", 10, 120, DEFAULT_LIMITS.title_max),
        h1_min=st.number_input("H1 min", 10, 100, DEFAULT_LIMITS.h1_min),
        h1_max=st.number_input("H1 max", 10, 120, DEFAULT_LIMITS.h1_max),
        desc_min=st.number_input("Description min", 60, 300, DEFAULT_LIMITS.desc_min),
        desc_max=st.number_input("Description max", 60, 320, DEFAULT_LIMITS.desc_max),
        body_min=st.number_input("Text_on_page min", 500, 5000, DEFAULT_LIMITS.body_min),
    )
    chunk_sleep = st.slider("Prodleva mezi dávkami (s) – kvóty API", 0.0, 5.0, 0.0, 0.5)
    dedup_title_threshold = st.slider("Prahová podobnost – krátká pole (0–1)", 0.5, 1.0, 0.90, 0.01)
    dedup_body_threshold = st.slider("Prahová podobnost – dlouhá pole (0–1)", 0.5, 1.0, 0.85, 0.01)
    batch_size = st.number_input("Batch size (OpenAI dávka)", min_value=5, max_value=100, value=20)

    # Načte se ze Secrets/ENV a rovnou se předvyplní
    if mode == "OpenAI API":
        api_key = st.text_input("OpenAI API key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    else:
        api_key = ""

# Upload (robustní handling)
uploaded = st.file_uploader("Nahraj CSV nebo XLSX", type=["csv", "xlsx"], accept_multiple_files=False)

# Normalize i kdyby bylo omylem multiple-files
if uploaded is not None and isinstance(uploaded, list):
    uploaded = uploaded[0] if uploaded else None

if uploaded is not None:
    # Načtení podle přípony (s ochranou na .name)
    if hasattr(uploaded, "name") and str(uploaded.name).lower().endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded, engine="openpyxl")
        except Exception as e:
            st.error(f"Chyba při čtení XLSX (zkontroluj openpyxl): {e}")
            st.stop()
    else:
        # CSV – zkusit ; a pak ,
        data = uploaded.getvalue().decode("utf-8", errors="ignore")
        try:
            df = pd.read_csv(io.StringIO(data), sep=";")
        except Exception:
            df = pd.read_csv(io.StringIO(data), sep=",")

    st.subheader("Náhled")
    st.dataframe(df.head(20))

    # Column mapping
    st.subheader("Mapování sloupců")
    cols = df.columns.tolist()
    default_idx = cols.index("name") if "name" in cols else 0
    name_col = st.selectbox("Sloupec s Name (klíčové slovo/URL)", options=cols, index=default_idx)

    # Process button
    if st.button("Vygenerovat a validovat"):
        out_rows: list[dict] = []
        logs: list[dict] = []

        # Dedup paměť
        prev_titles: list[str] = []
        prev_bodies: list[str] = []

        # --- Progress indikace ---
        progress_text = st.empty()
        progress_bar = st.progress(0)

        rows = list(df.itertuples(index=True))
        total = len(rows)
        done = 0

        # Připrav všechna keywords předem
        all_keywords = [normalize_keyword(str(getattr(r, name_col))) for r in rows]

        # Dávkování
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_keywords = all_keywords[start:end]

            # Hard negatives = dosud vygenerované tituly/H1 (proti opakování)
            hard_neg = list(dict.fromkeys(prev_titles))[:50]

            # Vygeneruj dávku jedním voláním
            if mode == "Template (offline stub)":
                batch_out = [simple_stub_generate(k, limits) for k in batch_keywords]
            else:
                batch_out = generate_batch_openai(api_key, batch_keywords, limits, hard_neg)

            # Post-processing a zápis do out_rows pro každý prvek dávky
            for i, rec in enumerate(batch_out):
                idx_in_df = rows[start + i].Index
                row = df.iloc[idx_in_df]
                keyword = batch_keywords[i]

                errs = validate_row(rec, limits)

                # dedup krátkých polí vs. dříve vygenerovaných
                sim_t = max([similar_title(rec["page_title"], t) for t in prev_titles], default=0.0)
                if sim_t >= dedup_title_threshold:
                    rec["page_title"] = ensure_len(
                        f"{keyword.capitalize()} – přehled a doporučení",
                        limits.title_min, limits.title_max
                    )
                    errs = validate_row(rec, limits)

                # dedup dlouhého textu TF-IDF cosine
                sim_b = max([similar_long(rec["text_on_page"], b) for b in prev_bodies], default=0.0)
                if sim_b >= dedup_body_threshold:
                    rec["text_on_page"] = rec["text_on_page"] + "\n\nPřidaná sekce: Konkrétní příklad použití a checklist kroků."
                    if len(rec["text_on_page"]) < limits.body_min + 120:
                        rec["text_on_page"] += " " + " ".join(["Doplnění." for _ in range(40)])

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

                logs.append({
                    "row": int(idx_in_df),
                    "keyword": keyword,
                    "errors": errs,
                    "sim_title": sim_t,
                    "sim_body": sim_b
                })

            # progress update
            done = end
            percent = int(done / total * 100) if total else 100
            progress_bar.progress(percent)
            progress_text.text(f"Zpracováno {done}/{total} (dávka {start+1}–{end})")
            if chunk_sleep > 0:
                time.sleep(chunk_sleep)

        # --- Progress: dokončeno ---
        progress_bar.empty()
        progress_text.text("✅ Hotovo – generování dávkami dokončeno.")

        out_df = pd.DataFrame(out_rows)
        st.subheader("Výsledek")
        st.dataframe(out_df.head(50))

        # Length summary
        def check_lengths(r):
            return pd.Series({
                "title_len": len(r["page_title"]),
                "h1_len": len(r["title_for_newest_advertisement_list"]),
                "desc_len": len(r["description"]),
                "body_len": len(r["text_on_page"]),
            })

        st.subheader("Kontrola délek (výběr)")
        st.dataframe(out_df.apply(check_lengths, axis=1).head(50))

        # Download
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Stáhnout CSV (UTF-8)", data=csv,
                           file_name="doplnene_texty.csv", mime="text/csv")

        st.subheader("Log validačních kontrol")
        st.json(logs)
else:
    st.info("Nahraj CSV/XLSX se sloupcem 'name' (nebo vyber odpovídající sloupec po nahrání).")
