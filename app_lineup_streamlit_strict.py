import io, re, unicodedata
from typing import List, Optional, Tuple, Set, Dict

import streamlit as st
import pandas as pd
from PIL import Image

import pytesseract
from pytesseract import TesseractNotFoundError

st.set_page_config(page_title="Excel ↔ Lineup Comparator (Strict)", layout="wide")
st.title("Excel ↔ Lineup Comparator (Strict)")
st.caption("OCRs a lineup screenshot and compares to Excel **exact-matching** normalized names. Extra junk text is filtered.")

BASE_STOPWORDS = {
    "home","farm","heart","sacred","lineup","timeline","info","table","h2h","matches",
    "substitution","substitutions","coach","g","c","lsl","fc","ireland",
    "vs","ft","ht","min","match","stadium","league","division","club",
    # common OCR shrapnel from UI
    "hi","im","sd","co","ney","info","table","lineup","h2h"
}

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'\s+', ' ', s)
    return s

def tokens_ok(lastname: str, stopwords: Set[str]) -> bool:
    ln = normalize_text(lastname)
    ln_clean = re.sub(r"[^a-z'\\-\\s]", "", ln)
    parts = [p for p in re.split(r"[\\s\\-]+", ln_clean) if p]
    if any(p in stopwords for p in parts):
        return False
    if not any(len(p) >= 3 for p in parts):
        return False
    return True

def name_to_last_initial(name: str, stopwords: Set[str]) -> Optional[str]:
    """Convert a name-like string into 'lastname f.' (lowercase), allowing compound lastnames."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = normalize_text(name)
    s = re.sub(r"[^a-z0-9'\\.\\-\\s]", "", s)

    # 1) "lastname parts... F."
    m = re.search(r"^([a-z][a-z '\\-]+)\\s+([a-z])\\.$", s)
    if m:
        last, initial = m.group(1).strip(), m.group(2)
        if tokens_ok(last, stopwords):
            return f"{last} {initial}."

    # 2) "F. lastname parts..."
    m = re.search(r"^([a-z])\\.\\s*([a-z][a-z '\\-]+)$", s)
    if m:
        initial, last = m.group(1), m.group(2).strip()
        if tokens_ok(last, stopwords):
            return f"{last} {initial}."

    # 3) "lastname, firstname"
    m = re.search(r"^([a-z][a-z '\\-]+)\\s*,\\s*([a-z]+)$", s)
    if m:
        last, first = m.group(1).strip(), m.group(2)
        if tokens_ok(last, stopwords):
            return f"{last} {first[0]}."

    # 4) "firstname ... lastname"
    parts = [p for p in re.split(r"\\s+", s) if p]
    if len(parts) >= 2:
        first = parts[0]
        last = " ".join(parts[1:])
        if tokens_ok(last, stopwords):
            return f"{last} {first[0]}."
    return None

def ensure_tesseract_available():
    try:
        _ = pytesseract.get_tesseract_version()
    except TesseractNotFoundError:
        st.error(
            "Tesseract OCR isn't installed or not on PATH.\n\n"
            "• **Streamlit Cloud**: add a `packages.txt` with `tesseract-ocr`\n"
            "• **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`\n"
            "• **macOS (Homebrew)**: `brew install tesseract`\n"
            "• **Windows**: install the UB Mannheim Tesseract build and add it to PATH"
        )
        st.stop()

def ocr_names(img: Image.Image, crop_box, stopwords: Set[str]) -> List[str]:
    ensure_tesseract_available()
    if crop_box:
        img = img.crop(crop_box)
    df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    if df is None or df.empty or "text" not in df.columns:
        return []
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    if df.empty:
        return []

    group_cols = ["page_num", "block_num", "par_num", "line_num"]
    lines = (
        df.sort_values(["page_num","block_num","par_num","line_num","word_num"])
          .groupby(group_cols)["text"]
          .apply(lambda toks: " ".join(str(t) for t in toks))
          .reset_index()["text"].tolist()
    )

    # Strict name chunks
    pattern = re.compile(r"[A-Za-z][A-Za-z'\\-]+\\s+[A-Z]\\.|[A-Z]\\.\\s*[A-Za-z][A-Za-z'\\-]+")
    candidates = set()
    for ln in lines:
        for m in pattern.finditer(ln):
            chunk = m.group(0)
            std = name_to_last_initial(chunk, stopwords)
            if std:
                candidates.add(std)
    return sorted(candidates)

def normalize_excel_names(df: pd.DataFrame, name_col: str, stopwords: Set[str]) -> Set[str]:
    if name_col not in df.columns:
        raise ValueError(f"Column '{name_col}' not found in Excel. Available: {list(df.columns)}")
    out = set()
    for v in df[name_col].astype(str).tolist():
        ni = name_to_last_initial(v, stopwords)
        if ni:
            out.add(ni)
    return out

# -------- UI --------
left, right = st.columns(2)
with left:
    excel_file = st.file_uploader("Excel/CSV roster", type=["xlsx","xls","csv"])
    name_col = st.text_input("Excel column with player names", value="player_name")
    extra_stops = st.text_input("Extra stopwords (comma-separated)", value="h2h, substitutions, matches")
with right:
    image_file = st.file_uploader("Lineup screenshot (PNG/JPG)", type=["png","jpg","jpeg"])
    st.caption("Crop to the two lineup columns only:")
    x0 = st.slider("Crop left %", 0, 40, 5)
    x1 = st.slider("Crop right %", 60, 100, 98)
    y0 = st.slider("Crop top %", 0, 40, 12)
    y1 = st.slider("Crop bottom %", 60, 100, 95)

run = st.button("Compare (strict exact-match)", type="primary", use_container_width=True)

if run:
    if excel_file is None or image_file is None:
        st.error("Please upload both the Excel/CSV and the screenshot image.")
        st.stop()

    # Full stopword set
    user_stops = {s.strip().lower() for s in extra_stops.split(",") if s.strip()}
    stopwords = BASE_STOPWORDS | user_stops

    # Load Excel
    if excel_file.name.lower().endswith(".csv"):
        df = pd.read_csv(excel_file, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(excel_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Normalize excel names
    excel_names = normalize_excel_names(df, name_col, stopwords)

    # OCR
    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    W, H = img.size
    crop_box = (int(W*x0/100), int(H*y0/100), int(W*x1/100), int(H*y1/100))
    ocr_candidates = ocr_names(img, crop_box, stopwords)

    # STRICT EXACT MATCH: keep only OCR names that exactly match an Excel normalized name
    ocr_exact = sorted(set(n for n in ocr_candidates if n in excel_names))

    # Possible OCR typos: same surname found, but initial doesn't match any Excel entry
    excel_by_surname: Dict[str, Set[str]] = {}
    for n in excel_names:
        last, initial = n.rsplit(" ", 1)
        excel_by_surname.setdefault(last, set()).add(initial)

    typos = []
    for n in ocr_candidates:
        last, initial = n.rsplit(" ", 1)
        if last in excel_by_surname and n not in excel_names:
            typos.append({"ocr": n, "excel_expected": sorted(excel_by_surname[last])})

    st.subheader("Excel normalized names")
    st.write(sorted(excel_names))

    st.subheader("OCR normalized names (exact matches only)")
    st.write(ocr_exact)

    # Mismatches (under strict mode)
    missing_on_website = sorted(excel_names - set(ocr_exact))
    extra_on_website = []  # by definition we don't report extras; strict mode only keeps known names

    c1, c2 = st.columns(2)
    with c1:
        st.metric("In Excel but not on website (strict)", len(missing_on_website))
        st.write(missing_on_website[:80])
    with c2:
        st.metric("On website but not in Excel (strict)", 0)
        st.caption("Strict mode suppresses extras by design. See 'Possible OCR typos' below.")

    # Possible OCR errors
    if typos:
        st.subheader("Possible OCR typos (same surname, different initial)")
        st.dataframe(pd.DataFrame(typos))

    # Downloads
    st.markdown("---")
    if missing_on_website:
        mo_df = pd.DataFrame({"missing_on_website": missing_on_website})
        st.download_button("Download 'missing on website' CSV",
                           data=mo_df.to_csv(index=False).encode("utf-8"),
                           file_name="missing_on_website.csv",
                           mime="text/csv")
