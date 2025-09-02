import io, re, unicodedata
from typing import List, Optional, Tuple, Set

import streamlit as st
import pandas as pd
from PIL import Image

# OCR
import pytesseract
from pytesseract import TesseractNotFoundError

st.set_page_config(page_title="Excel ↔ Website Lineup Comparator", layout="wide")
st.title("Excel ↔ Website Lineup Comparator")
st.caption("Upload your **Excel roster** and a **website screenshot (PNG/JPG)** of the lineup. I'll OCR the image, extract names, normalize them, and compare.")

with st.expander("What it does", expanded=False):
    st.markdown("""
- Extracts **player names** from the screenshot using OCR (Tesseract).
- Normalizes both sources to **`lastname f.`** (Last name + first initial) for robust matching.
- Compares sets: who is **in Excel but missing online**, and **online but not in Excel**.
""")

# ---------- Helpers ----------

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'\s+', ' ', s)
    return s

def name_to_last_initial(name: str) -> Optional[str]:
    """Convert a human name into 'lastname f.' format (lowercase, no accents)."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = normalize_text(name)
    s = re.sub(r"[^\w\s'\-\.]", "", s)

    m = re.search(r"([a-z][a-z'\-]+)\s+([a-z])\.", s)
    if m:
        last, initial = m.group(1), m.group(2)
        return f"{last} {initial}."

    m = re.search(r"([a-z])\.\s*([a-z][a-z'\-]+)", s)
    if m:
        initial, last = m.group(1), m.group(2)
        return f"{last} {initial}."

    m = re.search(r"([a-z][a-z'\-]+)\s*,\s*([a-z]+)", s)
    if m:
        last, first = m.group(1), m.group(2)
        return f"{last} {first[0]}."

    parts = [p for p in re.split(r"\s+", s) if p and p not in {"fc","lsl"}]
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        return f"{last} {first[0]}."
    return None

def ensure_tesseract_available():
    """Raise a clean, user-facing error if the Tesseract binary is missing."""
    try:
        _ = pytesseract.get_tesseract_version()
    except TesseractNotFoundError as e:
        st.error(
            "Tesseract OCR isn't installed or not on PATH.\n\n"
            "• **Streamlit Cloud**: add a `packages.txt` with `tesseract-ocr`\n"
            "• **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`\n"
            "• **macOS (Homebrew)**: `brew install tesseract`\n"
            "• **Windows**: install the UB Mannheim Tesseract build and add it to PATH"
        )
        st.stop()

def extract_names_from_image(img: Image.Image) -> List[str]:
    ensure_tesseract_available()
    df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    if df is None or df.empty or "text" not in df.columns:
        return []
    df = df.dropna(subset=["text"])  # remove NaNs
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
    candidates = set()
    for ln in lines:
        toks = re.findall(r"[A-Za-z][A-Za-z'\-\.]+", ln)
        for i in range(len(toks)):
            for j in range(i+1, min(i+4, len(toks))+1):
                chunk = " ".join(toks[i:j])
                std = name_to_last_initial(chunk)
                if std:
                    candidates.add(std)
    return sorted(candidates)

def to_last_initial_set_from_excel(df: pd.DataFrame, column: str) -> Set[str]:
    vals = df[column].astype(str).tolist()
    out = set()
    for v in vals:
        ni = name_to_last_initial(v)
        if ni:
            out.add(ni)
    return out

def read_excel(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(uploaded, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------- UI ----------
col1, col2 = st.columns(2)
with col1:
    excel_file = st.file_uploader("Excel/CSV roster", type=["xlsx","xls","csv"])
with col2:
    image_file = st.file_uploader("Website lineup screenshot (PNG/JPG)", type=["png","jpg","jpeg"])

if excel_file is not None:
    df = read_excel(excel_file)
    st.subheader("Pick the roster column from Excel")
    col_choices = list(df.columns)
    roster_col = st.selectbox("Roster column", options=col_choices, index=0 if col_choices else None)
else:
    df = None
    roster_col = None

run = st.button("Compare", type="primary", use_container_width=True)

if run:
    if df is None or image_file is None:
        st.error("Please upload both the Excel/CSV and the screenshot image.")
        st.stop()
    if not roster_col:
        st.error("Please select the roster column in your Excel/CSV.")
        st.stop()

    excel_names = to_last_initial_set_from_excel(df, roster_col)

    try:
        img = Image.open(image_file)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    ocr_names = set(extract_names_from_image(img))

    st.subheader("Normalized roster (Excel)")
    st.write(sorted(excel_names))

    st.subheader("Normalized lineup (Image OCR)")
    st.write(sorted(ocr_names))

    missing_on_website = sorted(excel_names - ocr_names)
    extra_on_website = sorted(ocr_names - excel_names)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("In Excel but not on website", len(missing_on_website))
        st.write(missing_on_website[:50])
    with c2:
        st.metric("On website but not in Excel", len(extra_on_website))
        st.write(extra_on_website[:50])

    st.markdown("---")
    if missing_on_website:
        mo_df = pd.DataFrame({"missing_on_website": missing_on_website})
        st.download_button("Download 'missing on website' CSV",
                           data=mo_df.to_csv(index=False).encode("utf-8"),
                           file_name="missing_on_website.csv",
                           mime="text/csv")
    if extra_on_website:
        eo_df = pd.DataFrame({"extra_on_website": extra_on_website})
        st.download_button("Download 'extra on website' CSV",
                           data=eo_df.to_csv(index=False).encode("utf-8"),
                           file_name="extra_on_website.csv",
                           mime="text/csv")
