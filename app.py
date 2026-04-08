import streamlit as st
import fitz
import easyocr
import numpy as np
import re
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import requests
import os
from dotenv import load_dotenv

# LOAD API KEYS & NLTK
load_dotenv()

@st.cache_resource
def download_nltk():
    nltk.download('stopwords')

download_nltk()
stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))

# HYBRID SERVER 
def is_lm_studio_online():
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=1)
        return response.status_code == 200
    except:
        return False

# Inisialisasi awal
client = None
model_name = ""

if is_lm_studio_online():
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    model_name = "qwen2.5-coder-3b-instruct"
    st.sidebar.success("🚀 Mode: Lokal (RTX 4050)")
else:
    try:
        groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except:
        groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        model_name = "llama-3.1-8b-instant"
        st.sidebar.info("⚡ Mode: Cloud (GROQ Engine)")
    else:
        st.sidebar.error("⚠️ API Key Groq Tidak Ditemukan!")

# TEXT PROCESSING FUNCTIONS
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    word_tokens = text.split()
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text)

def extract_text(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Trigger OCR
    if len(text.strip()) < 200:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = reader.readtext(np.array(img), detail=0)
            text += " ".join(result) + "\n"
    return text

@st.cache_resource
def load_ocr():
    # Tetap pake GPU=True
    return easyocr.Reader(['id', 'en'], gpu=True)

reader = load_ocr()

# UI STREAMLIT
st.set_page_config(page_title="ISHA AI - Career Coach", layout="wide")
st.title("🤖 ISHA AI - Career Coach")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Upload CV")
    uploaded_file = st.file_uploader("Pilih PDF CV", type=["pdf"])
    loker_input = st.text_area("Tempel Deskripsi Lowongan di sini:")

    if st.button("Analisis kecocokan skor"):
        if uploaded_file and loker_input:
            pdf_bytes = uploaded_file.getvalue()
            cv_raw = extract_text(pdf_bytes)
            st.session_state.cv_context = cv_raw 
            
            clean_cv = clean_text(cv_raw)
            clean_loker = clean_text(loker_input)
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([clean_cv, clean_loker])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            st.metric("Skor Kecocokan", f"{round(similarity * 100, 2)}%")
            st.progress(similarity)
        else:
            st.error("Upload CV & Isi Loker dulu!")

# Tampilkan Chat History
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# LOGIKA CHAT INTERAKTIF
if prompt := st.chat_input("Tanya apa aja ke Isha...."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ambil Konteks CV
    konteks_final = st.session_state.get("cv_context", "")
    
    with st.chat_message("assistant"):
        if not konteks_final:
            answer = "Bes, upload dulu dong CV-nya di sebelah kiri biar aku bisa baca dan kasih saran!"
        elif not client:
            answer = "Aduh Bes, aku gak punya akses ke otak AI-ku (API Key/Local Server mati)."
        else:
            try:
                response = client.chat.completions.create(
                    model=model_name, 
                    messages=[
                        {"role": "system", "content": f"Kamu adalah Isha, asisten karir yang empati. Gunakan data CV ini untuk membantu user: {konteks_final}"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Maaf Bes, server lagi ada kendala teknis: {e}"
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
