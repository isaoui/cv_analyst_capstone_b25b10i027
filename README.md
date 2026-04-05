# 🚀 SmartPath AI: Career Analysis & Recommendation System using RAG

**SmartPath AI** (atau Asisten Isha) adalah solusi berbasis Artificial Intelligence yang dirancang untuk membantu pelamar kerja melakukan audit CV secara mandiri. Dengan menggabungkan teknologi **Retrieval-Augmented Generation (RAG)** dan **Computer Vision**, sistem ini mampu memberikan analisis mendalam, skor kecocokan (*ATS Score*), serta rekomendasi peningkatan skill secara personal dan akurat.

---

## 🌟 Fitur Utama

- **Hybrid PDF Extraction Engine**: Menggabungkan **PyMuPDF** untuk ekstraksi teks digital dan **EasyOCR** untuk menangani PDF berbasis gambar (scan). Sistem secara otomatis mendeteksi jika dokumen memerlukan pemrosesan OCR.
- **RAG Architecture with ChromaDB**: Menggunakan **Vector Database (ChromaDB)** dengan embedding `all-MiniLM-L6-v2` untuk melakukan pencarian kemiripan kontekstual antara profil pengguna dan database karir.
- **Local LLM Integration (Qwen 2.5)**: Memanfaatkan model **Qwen 2.5-3B-Instruct** yang berjalan secara lokal via **LM Studio**. Data tetap aman di perangkat lokal tanpa perlu mengirim informasi sensitif ke cloud.
- **Career Gap Analysis**: Tidak hanya memberikan skor, asisten AI ini memberikan analisis kesenjangan (*gap*) antara skill saat ini dengan kebutuhan industri, lengkap dengan rekomendasi kursus spesifik.

---

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **LLM Engine**: LM Studio (OpenAI Compatible API)
- **Model LLM**: Qwen 2.5-3B-Instruct
- **Vector Database**: ChromaDB
- **OCR & Extraction**: EasyOCR, PyMuPDF (fitz)
- **Preprocessing**: NLTK, Regular Expression

---

## ⚙️ Persiapan & Instalasi

### 1. Prasyarat
- **LM Studio**: [Download & Install LM Studio](https://lmstudio.ai/).
- **Model**: Cari dan download model `Qwen2.5-3B-Instruct-GGUF`.
- **Server**: Nyalakan **Local Server** di LM Studio (Port Default: `1234`).

### 2. Instalasi Library
Pastikan kamu berada di folder proyek, lalu jalankan perintah berikut:
```bash
# pip install -r requirements.txt

Cara Penggunaan
Siapkan Data: Masukkan file CV kamu (format .pdf) ke dalam direktori proyek.

Jalankan Notebook: Buka NLP.ipynb menggunakan VS Code atau Jupyter Notebook.

Eksekusi Pipeline:

Jalankan sel inisialisasi library dan fungsi OCR.

Jalankan sel Ingestion untuk mendaftarkan dataset ke ChromaDB.

Pada sel terakhir, panggil fungsi analisis:

Python
analisis_cv_user("nama_cv_kamu.pdf", "deskripsi lowongan kerja")
Hasil: Asisten Isha akan menampilkan laporan lengkap di layar notebook.