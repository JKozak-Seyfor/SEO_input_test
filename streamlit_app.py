import os
import io
import re
import json
import time
from dataclasses import dataclass
import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    return s if s else ""


def ensure_len(text: str, min_len: int | None = None, max_len: int | None = None) -> str:
    t = (text or "").strip()
    if min_len and len(t) < min_len:
        t += " " + (" ".join(["…" for _ in range(min_len - len(t))]))
    if max_len and len(t) > max_len:
        t = t[:max_len]
        t = re.sub(r"\s+\S*$", "", t)
    return t


def simple_stub_generate(keyword: str, limits: Limits) -> dict:
    """Fallback, pokud selže API volání."""
    kw = (keyword or "Téma").capitalize()
    body = (
        f"<h3>{kw}</h3>\n"
        f"{kw} patří mezi témata, která lidé často hledají, ale zároveň u nich narážejí na rozporné informace. "
        f"V tomto přehledu shrnujeme klíčové pojmy, přínosy a omyly."
    )
    if len(body) < limits.body_min:
        body += " " + " ".join(["Rozšiřující text." for _ in range(80)])

    return {
        "page_title": ensure_len(f"{kw} – průvodce a tipy", limits.title_min, limits.title_max),
        "title_for_newest_advertisement_list": ensure_len(f"{kw} – novinky a přehled", limits.h1_min, limits.h1_max),
        "description": ensure_len(
            f"{kw} v kostce: co to je, k čemu slouží a jak z něj vytěžit maximum. Praktické rady a stručné tipy v jednom místě.",
            limits.desc_min, limits.desc_max,
        ),
        "text_on_page": ensure_len(body, min_len=limits.body_min),
        "page_title_eng": ensure_len(f"{kw} – guide and tips", limits.title_min, limits.title_max),
        "title_for_newest_advertisement_list_eng": ensure_len(f"{kw} – updates and overview", limits.h1_min, limits.h1_max),
        "description_eng": ensure_len(
            f"{kw} explained: what it is, where it helps and how to make the most of it. Practical advice and concise tips in one place.",
            limits.desc_min, limits.desc_max,
        ),
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
# OpenAI helpers
# -------------------------------

def _call_openai_json_safe(model_name: str, messages: list, max_completion_tokens: int = 6000, temperature: float = 0.6) -> str:
    """
    Zavolá ChatCompletion. Nejprve se pokusí o 'JSON mode' (response_format),
    pokud knihovna/model nepodporuje, zkusí standardní volání.
    Vrací textový obsah odpovědi.
    """
    # indikace do UI
    st.caption(f"🔌 volám OpenAI • model: **{model_name}** • json_mode: try")

    # 1) pokus s JSON mode
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # JSON mode – může, ale nemusí být podporovaný: pokud ne, vyhodí chybu a spadneme do fallbacku
            response_format={"type": "json_object"},
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e_json:
        st.warning(f"JSON mode není k dispozici nebo selhal: {e_json}. Pokračuji bez JSON mode.")
        # 2) standardní volání
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]


# -------------------------------
# OpenAI batch generator (GPT-5 nano / mini)
# -------------------------------

def generate_batch_openai(api_key: str, keywords: list[str], limits: Limits, hard_negatives: list[str], model_name: str) -> list[dict]:
    if openai is None or not api_key:
        return [simple_stub_generate(k, limits) for k in keywords]

    openai.api_key = api_key
    limits_payload = {
        "page_title": {"min": limits.title_min, "max": limits.title_max},
        "h1": {"min": limits.h1_min, "max": limits.h1_max},
        "description": {"min": limits.desc_min, "max": limits.desc_max},
        "text_on_page_min": limits.body_min,
    }

    # role/system
    sys_prompt = (
        "You are a skilled Czech SEO copywriter specializing in tasteful adult-oriented content. "
        "Generate unique, structured website texts in Czech and English exactly according to the JSON format provided. "
        "No exclamation marks, no vulgarities, no URLs. Titles and descriptions must fit the length limits strictly."
    )

    # vstupní data pro dávku
    items = [{"idx": i, "name": k} for i, k in enumerate(keywords)]

    # uživatelský prompt
    user_prompt = f"""
Máš pole objektů ({{idx, name}}):
{json.dumps(items, ensure_ascii=False)}

Pro každou položku vytvoř objekt se strukturou:
{{
  "idx": <číslo z inputu>,
  "page_title": "...",                         // CZ, 30–50 znaků, přirozeně obsahuje keyword (benefit/lákadlo), bez '!'
  "description": "...",                        // CZ, 140–160 znaků, shrnutí + jemné CTA, bez URL
  "text_on_page": "...",                       // CZ, ≥{limits.body_min} znaků, HTML-like: úvod → 2–3 H3 podtémata → závěr
  "title_for_newest_advertisement_list": "...",// CZ H1, 20–40 znaků, krátké, úderné, bez '!'
  "page_title_eng": "...",                     // EN varianta title, stejné limity
  "description_eng": "...",                    // EN varianta description, bez URL
  "text_on_page_eng": "...",                   // EN, zachovej strukturu, ≥{limits.body_min} znaků
  "title_for_newest_advertisement_list_eng": "..." // EN H1
}}

DŮLEŽITÉ:
• Každý objekt v poli odpovídá jednomu klíčovému slovu (name) a obsah MUSÍ být tematicky přizpůsoben tomuto názvu
  (např. „eroticke-masaze“, „privat“, „nocni-club“…).
• NEPOUŽÍVEJ generické fráze jako „patří mezi témata, která lidé často hledají“.
• Styl: smyslný, decentní, bez vulgarit.
• Vyhni se podobnostem s těmito titulky/H1 (hard negatives): {json.dumps(hard_negatives[:30], ensure_ascii=False)}

Odpověz pouze **validním JSON polem** bez úvodního textu, komentářů nebo vysvětlivek.
Každý prvek v poli musí odpovídat stejnému pořadí jako vstupní pole 'items'.

Délkové limity pro kontrolu:
{json.dumps(limits_payload, ensure_ascii=False, indent=2)}
"""

    # viditelná indikace pro uživatele
    st.info(f"📡 Odesílám dávku {len(keywords)} klíčových slov do OpenAI • model: **{model_name}**")

    try:
        content = _call_openai_json_safe(
            model_name=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
            temperature=0.6,
        )

        # očekáváme JSON pole
        start, end = content.find("["), content.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("Model nevrátil validní JSON pole (chybí '[' nebo ']').")

        arr = json.loads(content[start:end+1])
        out_by_idx = {int(x["idx"]): x for x in arr}

        results = []
        for i, k in enumerate(keywords):
            x = out_by_idx.get(i)
            results.append(simple_stub_generate(k, limits) if not x else {
                "page_title": x.get("page_title", ""),
                "title_for_newest_advertisement_list": x.get("title_for_newest_advertisement_list", ""),
                "description": x.get("description", ""),
                "text_on_page": x.get("text_on_page", ""),
                "page_title_eng": x.get("page_title_eng", ""),
                "title_for_newest_advertisement_list_eng": x.get("title_for_newest_advertisement_list_eng", ""),
                "description_eng": x.get("description_eng", ""),
                "text_on_page_eng": x.get("text_on_page_eng", ""),
            })

        st.success("✅ Odpověď z OpenAI přijata a zpracována.")
        return results

    except Exception as e:
        st.error(f"⚠️ Chyba při zpracování OpenAI odpovědi: {e}")
        if 'content' in locals():
            st.text_area("Částečný surový obsah odpovědi:", content[:4000])
        return [simple_stub_generate(k, limits) for k in keywords]


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="CSV Content Generator (GPT-5 nano/mini)", layout="wide")
st.title("CSV Content Generator (CZ/EN) — GPT-5 nano/mini")

with st.sidebar:
    st.header("Nastavení")
    mode = st.selectbox("Režim", ["Template (offline stub)", "OpenAI API"], index=1)

    # Přepínač modelu
    model_name = st.selectbox(
        "OpenAI model",
        options=["gpt-5-nano", "gpt-5-mini"],
        index=0,
        help="nano = levnější/rychlejší, mini = vyšší kvalita"
    )
    st.caption(f"Použitý model: **{model_name}**")

    batch_size = st.number_input("Batch size", 1, 20, 5)
    chunk_sleep = st.slider("Prodleva mezi dávkami (s)", 0.0, 5.0, 0.0, 0.5)
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.environ.get("OPENAI_API_KEY", ""))

uploaded = st.file_uploader("Nahraj CSV nebo XLSX", type=["csv", "xlsx"])
if uploaded is not None and isinstance(uploaded, list):
    uploaded = uploaded[0]

if uploaded:
    if hasattr(uploaded, "name") and uploaded.name.lower().endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded, engine="openpyxl")
        except Exception as e:
            st.error(f"Chyba při čtení XLSX: {e}")
            st.stop()
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

            if mode == "OpenAI API":
                batch_out = generate_batch_openai(api_key, batch_keywords, DEFAULT_LIMITS, hard_neg, model_name)
            else:
                batch_out = [simple_stub_generate(k, DEFAULT_LIMITS) for k in batch_keywords]

            for i, rec in enumerate(batch_out):
                idx_in_df = rows[start + i].Index
                row = df.iloc[idx_in_df]
                keyword = batch_keywords[i]

                errs = validate_row(rec, DEFAULT_LIMITS)

                prev_titles.append(rec["page_title"])
                prev_bodies.append(rec["text_on_page"])

                out_rows.append({"name": row.get(name_col, ""), **rec})
                logs.append({"row": int(idx_in_df), "keyword": keyword, "errors": errs})

            done = end
            progress_bar.progress(int(done / total * 100))
            progress_text.text(f"Zpracováno {done}/{total} (dávka {start+1}–{end})")

            if chunk_sleep > 0:
                time.sleep(chunk_sleep)

        progress_bar.empty()
        progress_text.text("✅ Hotovo – generování dávkami dokončeno.")

        out_df = pd.DataFrame(out_rows)
        st.subheader("Výsledek (náhled)")
        st.dataframe(out_df.head(50))

        st.download_button(
            "Stáhnout CSV (UTF-8)",
            out_df.to_csv(index=False).encode("utf-8"),
            "doplnene_texty.csv",
            mime="text/csv"
        )

        st.subheader("Log validačních kontrol")
        st.json(logs)
else:
    st.info("Nahraj CSV/XLSX se sloupcem 'name'.")
