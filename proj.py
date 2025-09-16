# Install first if not already:
# pip install streamlit PyPDF2 python-docx spacy scikit-learn matplotlib wordcloud

import streamlit as st
import PyPDF2
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# 📌 Helper Functions
# ---------------------------

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special chars
    return text

def get_keywords(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

def compute_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = (vectors * vectors.T).toarray()[0,1]  # cosine similarity

    return round(similarity * 100, 2)

# def generate_wordcloud(text):
#     wc = WordCloud(width=600, height=400, background_color="white").generate(text)
#     plt.figure(figsize=(6,4))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     st.pyplot(plt)

# ---------------------------
# 📌 Streamlit UI
# ---------------------------

st.title("📄 AI-Powered Resume Analyzer")
st.write("Upload your resume + job description → get ATS score & suggestions")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
jd_text = st.text_area("Paste Job Description here")

if resume_file and jd_text:
    # Extract resume text
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = extract_text_from_docx(resume_file)

    resume_text_clean = clean_text(resume_text)
    jd_text_clean = clean_text(jd_text)

    # Compute similarity (ATS Score)
    score = compute_similarity(resume_text_clean, jd_text_clean)
    st.subheader(f"📊 ATS Match Score: {score}%")

    # Extract keywords
    resume_keywords = set(get_keywords(resume_text_clean))
    jd_keywords = set(get_keywords(jd_text_clean))

    missing_keywords = jd_keywords - resume_keywords
    matched_keywords = jd_keywords & resume_keywords

    st.write("✅ Matched Keywords:", ", ".join(list(matched_keywords)[:15]))
    st.write("❌ Missing Keywords:", ", ".join(list(missing_keywords)[:15]))

    # Show Word Cloud
    # st.subheader("☁️ Resume Word Cloud")
    # generate_wordcloud(resume_text_clean)

    # Suggestions
    st.subheader("📝 Suggestions")
    if score < 50:
        st.warning("Low ATS score. Add more job-specific skills and keywords.")
    elif score < 75:
        st.info("Decent score, but you can still improve by adding more relevant experience.")
    else:
        st.success("Great! Your resume is highly relevant to this job.")

