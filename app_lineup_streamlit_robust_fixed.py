import io, re, unicodedata
from typing import List, Optional, Tuple, Set, Dict

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import pytesseract
from pytesseract import TesseractNotFoundError

st.set_page_config(page_title="Excel ↔ Lineup Comparator (Robust OCR, Fixed)", layout="wide")
st.title("Excel ↔ Lineup Comparator (Robust OCR, Fixed)")
st.caption("Fixes Tesseract config quoting; tries multiple OCR strategies and exact-matches to Excel.")

BASE_STOPWORDS = {
    "home","farm","heart","sacred","lineup","timeline","info","table","h2h","matches",
    "substitution","substitutions","coach","g","c","lsl","fc","ireland",
    "vs","ft","ht","min","match","stadium","league","division","club",
    "hi","im","sd","co","ney","info","table","lineup","h2h"
}

PSM_LIST = [6, 4, 11, 12, 7, 3, 13]
ROTATIONS = [0, 90, 270]

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
    if not isinstance(name, str) or not name.strip():
        return None
    s = normalize_text(name)
    s = re.sub(r"[^a-z0-9'\\.\\-\\s]", "", s)

    m = re.search(r"^([a-z][a-z '\\-]+)\\s+([a-z])\\.$", s)
    if m:
        last, initial = m.group(1).strip(), m.group(2)
        if tokens_ok(last, stopwords):
            return f"{last} {initial}."

    m = re.search(r"^([a-z])\\.\\s*([a-z][a-z '\\-]+)$", s)
    if m:
        initial, last = m.group(1), m.group(2).strip()
        if tokens_ok(last, stopwords):
            return f"{last} {initial}."

    m = re.search(r"^([a-z][a-z '\\-]+)\\s*,\\s*([a-z]+)$", s)
    if m:
        last, first = m.group(1).strip(), m.group(2)
        if tokens_ok(last, stopwords):
            return f"{last} {first[0]}."

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
            "Tesseract OCR isn't installed or not on PATH.\\n\\n"
            "• **Streamlit Cloud**: add a `packages.txt` with `tesseract-ocr`\\n"
            "• **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`\\n"
            "• **macOS (Homebrew)**: `brew install tesseract`\\n"
            "• **Windows**: install the UB Mannheim Tesseract build and add it to PATH"
        )
        st.stop()

def preprocess_variants(img, scale: float, invert: bool, sharpen: bool):
    base = img.convert("RGB")
    w, h = base.size
    if scale != 1.0:
        base = base.resize((int(w*scale), int(h*scale)))
    if invert:
        base = ImageOps.invert(base)
    gray = ImageOps.grayscale(base)
    if sharpen:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    varA = ImageOps.autocontrast(gray)
    varB = varA.point(lambda x: 255 if x > 180 else 0)
    varC = ImageOps.autocontrast(gray).filter(ImageFilter.MedianFilter(size=3))
    return [varA, varB, varC]

def ocr_try(img, psm: int):
    # FIX: quote the whitelist value so shlex doesn't choke on the apostrophe
    whitelist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.'- "
    cfg = f"--oem 1 --psm {psm} -l eng -c tessedit_char_whitelist=\"{whitelist}\""
    df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=cfg)
    if df is None or df.empty or "text" not in df.columns:
        return [], 0, 0
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    if df.empty:
        return [], 0, 0
    group_cols = ["page_num","block_num","par_num","line_num"]
    lines = (
        df.sort_values(["page_num","block_num","par_num","line_num","word_num"])
          .groupby(group_cols)["text"]
          .apply(lambda toks: " ".join(str(t) for t in toks))
          .reset_index()["text"].tolist()
    )
    token_count = int(df.shape[0])
    return lines, token_count, len(lines)

def try_all(img: Image.Image, crop_box, scale: float, invert: bool, sharpen: bool):
    ensure_tesseract_available()
    if crop_box:
        img = img.crop(crop_box)
    attempts = []
    for rot in [0, 90, 270]:
        rotated = img.rotate(rot, expand=True) if rot else img
        for pre in preprocess_variants(rotated, scale=scale, invert=invert, sharpen=sharpen):
            for psm in [6, 4, 11, 12, 7, 3, 13]:
                lines, toks, nlines = ocr_try(pre, psm)
                attempts.append({"rotation": rot, "psm": psm, "tokens": toks, "lines": nlines, "lines_text": lines})
    attempts.sort(key=lambda a: (a["tokens"], a["lines"]), reverse=True)
    return attempts

def extract_candidates_from_lines(lines: List[str], stopwords: Set[str]) -> List[str]:
    candidates = set()
    for ln in lines:
        toks = re.findall(r"[A-Za-z][A-Za-z'\\-\\.]+", ln)
        for i in range(len(toks)):
            for j in range(i+1, min(i+4, len(toks))+1):
                chunk = " ".join(toks[i:j])
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

advanced = st.expander("OCR Advanced", expanded=False)
with advanced:
    scale = st.slider("Upscale image", 1.0, 3.0, 1.8, step=0.1)
    invert = st.checkbox("Invert colors", value=False)
    sharpen = st.checkbox("Sharpen", value=True)
    show_attempts = st.checkbox("Show best attempt details", value=True)
    show_lines = st.checkbox("Show OCR lines", value=False)
    show_candidates = st.checkbox("Show OCR candidates", value=False)

run = st.button("Compare (Robust, Fixed)", type="primary", use_container_width=True)

if run:
    if excel_file is None or image_file is None:
        st.error("Please upload both the Excel/CSV and the screenshot image.")
        st.stop()

    user_stops = {s.strip().lower() for s in extra_stops.split(",") if s.strip()}
    stopwords = BASE_STOPWORDS | user_stops

    if excel_file.name.lower().endswith(".csv"):
        df = pd.read_csv(excel_file, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(excel_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    excel_names = normalize_excel_names(df, name_col, stopwords)

    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    W, H = img.size
    crop_box = (int(W*x0/100), int(H*y0/100), int(W*x1/100), int(H*y1/100))
    attempts = try_all(img, crop_box, scale, invert, sharpen)

    if not attempts:
        st.error("OCR made no progress. Try widening the crop, increasing upscale, or toggling invert.")
        st.stop()

    best = attempts[0]
    if show_attempts:
        st.subheader("Best OCR attempt")
        st.write({"rotation": best["rotation"], "psm": best["psm"], "tokens": best["tokens"], "lines": best["lines"]})

    lines = best["lines_text"]

    if show_lines:
        st.subheader("OCR lines (best attempt)")
        st.write(lines)

    candidates = extract_candidates_from_lines(lines, stopwords)
    if show_candidates:
        st.subheader("OCR candidates (normalized)")
        st.write(candidates)

    exact_matches = sorted(set(n for n in candidates if n in excel_names))

    st.subheader("Excel normalized names")
    st.write(sorted(excel_names))

    st.subheader("OCR normalized names (exact matches)")
    st.write(exact_matches)

    missing_on_website = sorted(excel_names - set(exact_matches))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("In Excel but not on website", len(missing_on_website))
        st.write(missing_on_website[:80])
    with c2:
        st.metric("On website but not in Excel", 0)
        st.caption("Exact-match mode suppresses extras.")

    st.markdown("---")
    if missing_on_website:
        mo_df = pd.DataFrame({"missing_on_website": missing_on_website})
        st.download_button("Download 'missing on website' CSV",
                           data=mo_df.to_csv(index=False).encode("utf-8"),
                           file_name="missing_on_website.csv",
                           mime="text/csv")
