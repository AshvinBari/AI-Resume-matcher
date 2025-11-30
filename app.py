import streamlit as st
import pdfplumber
import re
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from streamlit_lottie import st_lottie
import os
import json

# --- Page Config ---
st.set_page_config(page_title="Resume Matcher", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        h1, h3, h4 {text-align: center;}
        .card {
            background-color: #060606;
            padding: 20px;
            margin: 10px 0;
            border-radius: 15px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
        .info-block {
            padding: 10px 15px;
            # background-color: #000000;
            border-radius: 10px;
            font-size: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Download spaCy model ---
# --- Download spaCy model (Handled in requirements.txt) ---
# os.system("python -m spacy download en_core_web_sm")

# --- Load models ---
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-mpnet-base-v2")
    nlp = spacy.load("en_core_web_sm")
    kw_model = KeyBERT(model=model)
    return model, nlp, kw_model

model, nlp, kw_model = load_models()

# --- Load animation ---
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_match = load_lottiefile("D:\Resume_ATS\data\Animation.json")

# --- Constants ---
DEGREE_KEYWORDS = ["Bachelor", "B.Tech", "M.Tech", "PhD", "Diploma"]
COMPANY_KEYWORDS = ["Technologies", "Solutions", "Labs", "Inc", "LLC"]
COLLEGE_KEYWORDS = ["Institute", "University", "College"]

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text.strip()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_section(text, section_name):
    pattern = re.compile(rf'{section_name}(.+?)(?=(Education|Experience|Skills|Projects|Certifications|$))',
                         re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""

def extract_skills_auto(text, top_n=15):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

def calculate_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score * 100, 2)

def extract_degrees(text):
    return list(set([deg for deg in DEGREE_KEYWORDS if re.search(rf"\b{deg}\b", text, re.IGNORECASE)]))

def extract_organizations(text):
    lines = text.split('\n')
    experience_orgs, education_orgs = set(), set()
    for line in lines:
        if any(word in line for word in COLLEGE_KEYWORDS):
            education_orgs.add(line.strip())
        elif any(word in line for word in COMPANY_KEYWORDS):
            experience_orgs.add(line.strip())
    return list(experience_orgs), list(education_orgs)

def enhanced_info_extraction(resume_text, edu_text, exp_text):
    degrees = extract_degrees(edu_text + ' ' + resume_text)
    exp_orgs, edu_orgs = extract_organizations(edu_text + '\n' + exp_text)
    return degrees, list(set(exp_orgs + edu_orgs))

# --- Header with Animation ---
if lottie_match:
    st_lottie(lottie_match, height=120, key="match")

st.markdown("<h1>🤖 AI Resume Matcher</h1>", unsafe_allow_html=True)
st.markdown("<h4>Upload your resume and match it with job descriptions instantly</h4>", unsafe_allow_html=True)

# --- Upload & Input Section (Mobile Friendly) ---
with st.container():
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader("Upload your PDF resume", type=["pdf"])

    st.markdown("### 📝 Job Description")
    job_description = st.text_area("Paste JD here", height=250)

# --- Match Button ---
if st.button("🔍 Match Now", use_container_width=True):
    if resume_file and job_description:
        resume_text = clean_text(extract_text_from_pdf(resume_file))
        edu_text = extract_section(resume_text, "Education")
        exp_text = extract_section(resume_text, "Experience")

        resume_skills = extract_skills_auto(resume_text)
        jd_skills = extract_skills_auto(job_description)
        degrees, orgs = enhanced_info_extraction(resume_text, edu_text, exp_text)
        score = calculate_similarity(" ".join(resume_skills), " ".join(jd_skills))

        st.divider()

        # Match Score
        st.markdown(f"<h3>✅ Match Score: <span style='color:#2196f3;'>{score}%</span></h3>", unsafe_allow_html=True)

        bar_color = "#4caf50" if score > 85 else "#ffc107" if score > 65 else "#ff9800" if score > 45 else "#f44336"
        st.markdown(f"""
            <div style="background-color: #ddd; border-radius: 10px; height: 20px; width: 100%;">
                <div style="width: {score}%; background-color: {bar_color}; height: 100%; border-radius: 10px;"></div>
            </div>
            <p style="text-align:center; margin-top:5px; font-weight:bold;">{score}% Match</p>
        """, unsafe_allow_html=True)

        # Extracted Info Section
        st.markdown("## 🔍 Extracted Information")

        with st.expander("📌 Required Skills", expanded=True):
            st.markdown(f"<div class='card'><div class='info-block'>{', '.join(jd_skills)}</div></div>", unsafe_allow_html=True)

        with st.expander("🛠️ Resume Skills", expanded=True):
            st.markdown(f"<div class='card'><div class='info-block'>{', '.join(resume_skills)}</div></div>", unsafe_allow_html=True)

        with st.expander("🎓 Degrees Found", expanded=False):
            st.markdown(f"<div class='card'><div class='info-block'>{', '.join(degrees) if degrees else 'Not Found'}</div></div>", unsafe_allow_html=True)

        with st.expander("💼 Organizations Mentioned", expanded=False):
            st.markdown(f"<div class='card'><div class='info-block'>{', '.join(orgs) if orgs else 'Not Found'}</div></div>", unsafe_allow_html=True)

    else:
        st.warning("📢 Please upload a resume and paste a job description.")

# --- Footer ---
st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 14px;">
        Developed by <strong>Ashvinkumar Bari</strong> • Powered by <code>Streamlit</code> & <code>KeyBERT</code>
    </p>
""", unsafe_allow_html=True)
