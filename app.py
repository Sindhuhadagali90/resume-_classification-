# app.py (final fixed version)

import io
import os
import re
import tempfile
import pickle
import pandas as pd
import streamlit as st

# NLP libraries
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# File parsing
import pdfplumber
import docx2txt
import subprocess
import sys

# -------------------------
# Ensure NLTK data is available
def ensure_nltk_data():
    needed = ["wordnet", "omw-1.4", "punkt", "stopwords"]
    for pkg in needed:
        try:
            if pkg == "punkt":
                nltk.data.find(f"tokenizers/{pkg}")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)
ensure_nltk_data()

# -------------------------
# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
    )
    nlp = spacy.load("en_core_web_sm")


# Precompute stopwords
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

# -------------------------
st.title("RESUME CLASSIFICATION")
st.markdown("<style>h1{color: Purple;}</style>", unsafe_allow_html=True)
st.subheader("Welcome to Resume Classification App")

# ---------------------------------------------------------
# Updated SKILLS.CSV loader (no IndexError)
SKILLS_CSV_PATH = "skills.csv"
skills_list = []

if not os.path.exists(SKILLS_CSV_PATH):
    st.warning(f"Skills file not found at '{SKILLS_CSV_PATH}'. extract_skills will return empty lists.")
else:
    try:
        skills_df = pd.read_csv(SKILLS_CSV_PATH, header=0, dtype=str)

        if not skills_df.empty:
            # Add column headers
            headers = [str(h).strip() for h in skills_df.columns.tolist() if str(h).strip()]

            # Add any non-empty cell values
            cell_values = []
            for col in skills_df.columns:
                col_vals = skills_df[col].dropna().astype(str).str.strip().tolist()
                cell_values.extend([v for v in col_vals if v])

            combined = set([s.lower() for s in headers + cell_values])
            skills_list = sorted([s for s in combined if s])
        else:
            skills_list = []
    except Exception as e:
        st.warning(f"Could not read skills CSV: {e}. extract_skills will return empty lists.")
        skills_list = []

skills_list = [s.strip().lower() for s in skills_list if isinstance(s, str) and s.strip()]

# ---------------------------------------------------------
def extract_skills(resume_text: str):
    if not resume_text:
        return []
    nlp_text = nlp(resume_text)
    noun_chunks = [chunk.text.lower().strip() for chunk in nlp_text.noun_chunks]
    tokens = [token.text.lower() for token in nlp_text if not token.is_stop]

    skillset = []

    for token in tokens:
        if token in skills_list:
            skillset.append(token)

    for nc in noun_chunks:
        if nc in skills_list:
            skillset.append(nc)

    return [s.capitalize() for s in sorted(set(skillset))]


# ---------------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text()
            if ptext:
                text.append(ptext)
    return "\n".join(text)


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(docx_bytes)
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path) or ""
    finally:
        os.remove(tmp_path)


def get_text_from_uploaded(uploaded_file) -> str:
    uploaded_file.seek(0)
    data = uploaded_file.read()

    mime = uploaded_file.type or ""
    name = uploaded_file.name.lower()

    if "word" in mime or name.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    elif "pdf" in mime or name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    else:
        # fallback
        try:
            return extract_text_from_pdf_bytes(data)
        except:
            return extract_text_from_docx_bytes(data)


# ---------------------------------------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)

    tokens = tokenizer.tokenize(text)

    filtered = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemma_words)

# ---------------------------------------------------------
# Load model and vectorizer
try:
    model = pickle.load(open("modelDT.pkl", "rb"))
    vectorizer = pickle.load(open("vector.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.info("Please run 'python train_model.py' first to generate the model files.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model/vectorizer: {e}")
    st.stop()

# ---------------------------------------------------------
upload_files = st.file_uploader("Upload Your Resumes", type=["docx", "pdf"], accept_multiple_files=True)

results = []

if upload_files:
    for uploaded in upload_files:
        try:
            text = get_text_from_uploaded(uploaded)
            cleaned = preprocess(text)
            prediction = model.predict(vectorizer.transform([cleaned]))[0]
            skill_found = extract_skills(text)

            results.append({
                "Uploaded File": uploaded.name,
                "Predicted Profile": prediction,
                "Skills": ", ".join(skill_found)
            })
        except Exception as e:
            results.append({
                "Uploaded File": uploaded.name,
                "Predicted Profile": f"Error: {e}",
                "Skills": ""
            })

# Display full table
if results:
    df = pd.DataFrame(results)
    st.table(df)

    select = ["PeopleSoft", "SQL Developer", "React JS Developer", "Workday"]
    st.subheader("Select as per Requirement")

    option = st.selectbox("Fields", select)

    if option:
        st.table(df[df["Predicted Profile"] == option])
else:
    st.info("Upload one or more resumes to see predictions.")



