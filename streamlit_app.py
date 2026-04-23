import streamlit as st
import requests
import io
from docx import Document

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEO Brief Generator",
    page_icon="📝",
    layout="centered",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { max-width: 680px; padding-top: 2.5rem; }
    .stTextInput > label, .stTextArea > label,
    .stSelectbox > label, .stFileUploader > label {
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Webhook URL ze Streamlit secrets ─────────────────────────────────────────
WEBHOOK_URL = st.secrets.get("WEBHOOK_URL", "")

# ── Klienti ───────────────────────────────────────────────────────────────────
CLIENTS = ["mBank"]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("SEO Brief Generator")
st.markdown("Zadej téma článku a agent automaticky sestaví keyword brief.")
st.divider()

client_id = st.selectbox(
    "Klient",
    options=CLIENTS,
    help="Vyber klienta – nastaví keyword pravidla a brand kontext.",
)

topic = st.text_input(
    "Téma článku",
    placeholder="např. Švarcsystém: co to je, kdy hrozí pokuta a jak se mu vyhnout",
    help="Napiš téma co nejkonkrétněji – agent z něj odvodí seed keywords.",
)

notes = st.text_area(
    "Poznámky (volitelné)",
    placeholder="např. rozšíření existujícího článku, zaměření na OSVČ, délka ~2000 slov…",
    height=100,
)

uploaded_file = st.file_uploader(
    "Existující článek (volitelné)",
    type=["docx"],
    help="Nahraj .docx, pokud jde o rozšíření nebo přepracování existujícího textu.",
)

st.divider()
submit = st.button("Spustit agenta →", type="primary", use_container_width=True)

# ── Logika odeslání ───────────────────────────────────────────────────────────
if submit:
    # Validace
    if not topic.strip():
        st.error("Téma článku je povinné.")
        st.stop()

    if not WEBHOOK_URL:
        st.error("WEBHOOK_URL není nastavena v secrets.toml.")
        st.stop()

    # Extrakce textu z docx
    existing_article = ""
    if uploaded_file is not None:
        try:
            doc = Document(io.BytesIO(uploaded_file.read()))
            existing_article = "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )
        except Exception as e:
            st.error(f"Nepodařilo se načíst soubor: {e}")
            st.stop()

    # Sestavení payloadu
    payload = {
        "client_id": client_id,
        "topic": topic.strip(),
        "notes": notes.strip(),
        "existing_article": existing_article,
    }

    # Odeslání na webhook
    with st.spinner("Odesílám na agenta…"):
        try:
            response = requests.post(WEBHOOK_URL, json=payload, timeout=30)
            response.raise_for_status()
            st.success("✅ Agent spuštěn. Brief bude brzy k dispozici.")
            st.caption(f"Status: {response.status_code} · Klient: {client_id} · Téma: {topic}")
        except requests.exceptions.Timeout:
            st.error("Webhook neodpověděl včas (timeout 30 s). Zkus to znovu.")
        except requests.exceptions.RequestException as e:
            st.error(f"Chyba při odesílání: {e}")
