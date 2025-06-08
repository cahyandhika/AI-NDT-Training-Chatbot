# Chatbot RAG dengan Streamlit

## 📌 Deskripsi Proyek
Proyek ini adalah chatbot yang menggunakan teknik **Retrieval-Augmented Generation (RAG)** untuk memberikan jawaban berbasis fakta dan informasi statistik. Chatbot ini memanfaatkan **Groq LLM** untuk pemrosesan bahasa, **Pinecone** untuk pencarian vektor, serta **Streamlit** sebagai antarmuka pengguna.

## 🚀 Fitur Utama
- **Pencarian Kontekstual** → Menggunakan Pinecone untuk mencari informasi yang relevan.
- **Pemrosesan Bahasa Alami** → Memanfaatkan model bahasa dari Groq untuk menghasilkan jawaban.
- **Antarmuka Interaktif** → Dibangun dengan Streamlit untuk pengalaman pengguna yang mudah dan menarik.
- **Penggunaan Statistik dan Fakta** → Jawaban chatbot didasarkan pada data nyata.

## 🔧 Instalasi
Pastikan Anda sudah menginstal **Python 3.8+** dan berikut paket yang diperlukan:

```bash
pip install streamlit langchain pinecone-client groq python-dotenv

Cara Menjalankan Aplikasi
- Clone repositori ini atau simpan file proyek di komputer Anda.
- Siapkan variabel lingkungan di file .env:
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

- buat virtual environment dengan cara: 
python -m venv .venv

- lalu masuk kedalam virtual environment dengan cara:
.venv\Scripts\activate #windows users

- install semua dependensi yang ada di requirements.txt dengan cara:
pip install -r requirements.txt

- Jalankan aplikasi Streamlit:
streamlit run app.py


🛠️ Struktur Proyek
📂 AI NDT Training Chatbot
├── 📜 app.py             # Skrip utama Streamlit
├── 📜 rag_generation_pc.py # Implementasi pipeline RAG
├── 📜 requirements.txt   # Daftar dependensi
├── 📜 .env               # Variabel lingkungan API
└── 📂 data               # Folder untuk data yang digunakan


⚙️ Cara Kerja
- Pengguna memasukkan pertanyaan melalui antarmuka Streamlit.
- Pinecone mencari informasi terkait berdasarkan embedding teks.
- Groq LLM menghasilkan jawaban berdasarkan konteks yang ditemukan.
- Jawaban ditampilkan secara interaktif kepada pengguna.

📢 Kontribusi
Jika Anda ingin mengembangkan proyek ini lebih lanjut, silakan buat pull request atau buka issue di repositori ini!

🔥 Selamat menggunakan chatbot RAG ini! 🚀

README ini menjelaskan secara lengkap proyek chatbot, termasuk instalasi, cara kerja, dan fitur utama. Semoga membantu! Jika ada bagian yang ingin ditambahkan atau diperjelas, beri tahu saya. 😊