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
    """Fallback, pokud selže API volání."""
    kw = keyword.capitalize() or "Téma"
    body = f"<h3>{kw}</h3>\n{kw} patří mezi témata, která lidé často hledají, ale zároveň u nich narážejí na rozporné informace."
    if len(body) < limits.body_min:
        body += " " + " ".join(["Rozšiřující text." for _ in range(80)])

    return {
        "page_title": ensure_len(f"{kw} – průvodce a tipy", limits.title_min, limits.title_max),
        "title_for_newest_advertisement_list": ensure_len(f"{kw} – novinky a přehled", limits.h1_min, limits.h1_max),
        "description": ensure_len(
            f"{kw} v kostce: co to je, k čemu slouží a jak z něj vytěžit maximum. "
            f"Praktické rady a stručné tipy v jednom místě.",
            limits.desc_min, limits.desc_max,
        ),
        "text_on_page": ensure_len(body, min_len=limits.body_min),
        "page_title_eng": ensure_len(f"{kw} – guide and tips", limits.title_min, limits.title_max),
        "title_for_newest_advertisement_list_eng": ensure_len(f"{kw} – updates and overview", limits.h1_min, limits.h1_max),
        "description_eng": ensure_len(
            f"{kw} explained: what it is, where it helps and how to make the most of it. "
            f"Practical advice and concise tips in one place.",
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
    errs =
