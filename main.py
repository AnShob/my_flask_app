import streamlit as st
import pandas as pd
from joblib import load
import nltk
import os
from preprosesing import Preprocessor

# Menentukan lokasi data NLTK
# Mendapatkan path direktori file skrip yang sedang dijalankan
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path ke folder 'punkt'
nltk_data_path = os.path.join(current_dir, 'punkt')

# Menambahkan path ke NLTK
nltk.data.path.append(nltk_data_path)

# Pastikan dataset NLTK tersedia
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

# Memuat model dan vectorizer
loaded_model = load('svm_model(RBF).joblib')
loaded_vectorizer = load('tfidf_vectorizer.joblib')

# Instance preprocessor
preprocessor = Preprocessor(sinopsis_column='Sinopsis')

# Streamlit App
st.title("Klasifikasi Genre Buku")
st.write("Masukkan sinopsis buku untuk memprediksi genre.")

sinopsis = st.text_area("Masukkan Sinopsis:")

if st.button("Prediksi Genre"):
    if sinopsis.strip():
        # Membuat DataFrame dari input pengguna
        data_baru = pd.DataFrame({'Sinopsis': [sinopsis]})
        st.write("hai")
        
        # Preprocessing data baru
        try:
            st.write("Input DataFrame sebelum preprocessing:", data_baru)
            try:
                data_baru = preprocessor.preprocess_dataframe(data_baru)
                st.write("Setelah Preprocessing:", data_baru)
            except Exception as e:
                st.error(f"Terjadi error saat preprocessing: {e}")
                st.stop()

            # sinopsis_tfidf = loaded_vectorizer.transform(data_baru['Lemmatized'].apply(' '.join))
            sinopsis_tfidf = loaded_vectorizer.transform(data_baru)
            st.write("TF-IDF Shape:", sinopsis_tfidf.shape)
        
            prediksi = loaded_model.predict(sinopsis_tfidf.toarray())
            st.success(f"Genre yang diprediksi: **{prediksi[0]}**")
        except Exception as e:
            st.write("Checkpoint: Preprocessor instance dibuat")


