# app.py 
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
# Load lightweight spaCy pipeline (Streamlit Cloud safe)
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Precompute stopwords
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

# -------------------------
st.title("RESUME CLASSIFICATION")
st.markdown("<style>h1{color: Purple;}</style>", unsafe_allow_html=True)
st.subheader("Welcome to Resume Classification App")

# ---------------------------------------------------------
# Skills CSV loader
SKILLS_CSV_PATH = "skills.csv"
skills_list = []

if os.path.exists(SKILLS_CSV_PATH):
    try:
        skills_df = pd.read_csv(SKILLS_CSV_PATH, dtype=str)
        headers = [h.lower().strip() for h in skills_df.columns if h]
        cells = skills_df.fillna("").values.flatten().tolist()
        skills_list = sorted(set(headers + [c.lower().strip() for c in cells if c]))
    except Exception:
        skills_list = []

# ---------------------------------------------------------
def extract_skills(resume_text: str):
    if not resume_text:
        return []

    doc = nlp(resume_text.lower())
    tokens = [t.text for t in doc if not t.is_stop]
    skills = set()

    for token in tokens:
        if token in skills_list:
            skills.add(token.capitalize())

    return sorted(skills)

# ---------------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text.append(page.extract_text())
    return "\n".join(text)

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(docx_bytes)
        path = tmp.name
    try:
        return docx2txt.process(path) or ""
    finally:
        os.remove(path)

def get_text_from_uploaded(uploaded_file) -> str:
    data = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    return extract_text_from_docx_bytes(data)

# ---------------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ---------------------------------------------------------
# Load model and vectorizer
try:
    model = pickle.load(open("modelDT.pkl", "rb"))
    vectorizer = pickle.load(open("vector.pkl", "rb"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------------
upload_files = st.file_uploader(
    "Upload Your Resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

results = []

if upload_files:
    for uploaded in upload_files:
        try:
            raw_text = get_text_from_uploaded(uploaded)
            cleaned = preprocess(raw_text)
            prediction = model.predict(vectorizer.transform([cleaned]))[0]
            skills = extract_skills(raw_text)

            results.append({
                "File Name": uploaded.name,
                "Predicted Role": prediction,
                "Skills": ", ".join(skills)
            })
        except Exception as e:
            results.append({
                "File Name": uploaded.name,
                "Predicted Role": "Error",
                "Skills": str(e)
            })

if results:
    df = pd.DataFrame(results)
    st.table(df)

    st.subheader("Filter by Role")
    role = st.selectbox("Select Role", df["Predicted Role"].unique())
    st.table(df[df["Predicted Role"] == role])
else:
    st.info("Upload resumes to see predictions.")
