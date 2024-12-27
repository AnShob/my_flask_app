import streamlit as st
import pandas as pd
from joblib import load
from preprosesing import Preprocessor
import nltk

# Download stopwords jika belum tersedia
nltk.download('stopwords')
nltk.download('punkt')

# Memuat model SVM
loaded_model = load('svm_model(RBF).joblib')

# Memuat TF-IDF Vectorizer
loaded_vectorizer = load('tfidf_vectorizer.joblib')

# Instance preprocessor
preprocessor = Preprocessor(sinopsis_column='Sinopsis')

# Judul Aplikasi
st.title("Klasifikasi Genre Buku Berdasarkan Sinopsis")
st.write("Masukkan sinopsis buku untuk memprediksi genre.")

# Input sinopsis dari pengguna
sinopsis = st.text_area("Masukkan Sinopsis Buku di bawah ini:")

# Tombol untuk prediksi
if st.button("Prediksi Genre"):
    if sinopsis.strip():
        # Membuat DataFrame untuk sinopsis baru
        data_baru = pd.DataFrame({'Sinopsis': [sinopsis]})

        # Preprocessing data baru
        data_baru_processed = preprocessor.preprocess_dataframe(data_baru)

        # Transform sinopsis menjadi TF-IDF
        sinopsis_tfidf = loaded_vectorizer.transform(data_baru_processed['Lemmatized'].apply(' '.join))
        sinopsis_dense = sinopsis_tfidf.toarray()  # Ubah sparse matrix menjadi dense array

        # Prediksi dengan model
        prediksi = loaded_model.predict(sinopsis_dense)
        genre_prediksi = prediksi[0]  # Ambil genre pertama dari hasil prediksi

        # Tampilkan hasil prediksi
        st.success(f"Genre yang diprediksi adalah: **{genre_prediksi}**")
    else:
        st.error("Silakan masukkan sinopsis buku!")
