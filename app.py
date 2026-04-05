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
import chromadb

# NLTK
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')

download_nltk()
stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    word_tokens = text.split()
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Database
chroma_client = chromadb.PersistentClient(path="db_smartpath")
collection = chroma_client.get_or_create_collection(name="cv_collection")
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def extract_text(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    if len(text.strip()) < 200:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = reader.readtext(np.array(img), detail=0)
            text += " ".join(result) + "\n"
    return text

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['id', 'en'], gpu=True)

reader = load_ocr()

# UI Streamlit
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
            cv_raw = extract_text(uploaded_file.read())
            st.session_state.cv_context = cv_raw 
            
            st.write("DEBUG: Karakter kebaca:", len(cv_raw))
            with st.expander("Lihat teks yang diekstrak"):
                st.write(cv_raw[:1000])
            
            clean_cv = clean_text(cv_raw)
            clean_loker = clean_text(loker_input)
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([clean_cv, clean_loker])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            st.metric("Skor Kecocokan", f"{round(similarity * 100, 2)}%")
            st.progress(similarity)
        else:
            st.error("Upload CV & Isi Loker dulu!")

# Tampilkan Chat
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat
if prompt := st.chat_input("Tanya apa aja ke Isha...."):
    # Tampilkan Chat User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ambil Konteks CV
    konteks_final = ""
    if "cv_context" in st.session_state:
        konteks_final = st.session_state.cv_context
    elif uploaded_file is not None:
        # Kalau tombol analisis belum diklik tapi file udah ada, kita baca paksa
        konteks_final = extract_text(uploaded_file.getvalue())
        st.session_state.cv_context = konteks_final
    
    # Tanya AI
    with st.chat_message("assistant"):
        if konteks_final == "":
            answer = "Bes, upload dulu dong CV-nya di sebelah kiri biar aku bisa baca dan kasih saran!"
        else:
            try:
                response = client.chat.completions.create(
                    model="model-identifier",
                    messages=[
                        {"role": "system", "content": f"Kamu adalah Isha, asisten karir. Gunakan data CV ini untuk menjawab: {konteks_final}"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Aduh Bes, ada kabel putus ke LM Studio: {e}"
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})