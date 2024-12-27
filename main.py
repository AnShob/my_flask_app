import streamlit as st
import pandas as pd
from joblib import load
import nltk
import os
from preprosesing import Preprocessor

# Menentukan lokasi data NLTK
nltk_data_path = os.path.join(os.path.dirname(__file__), 'punkt')
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
            st.write("Input DataFrame:", data_baru)
            data_baru_processed = preprocessor.preprocess_dataframe(data_baru)
            st.write("Setelah Preprocessing:", data_baru_processed)
        
            sinopsis_tfidf = loaded_vectorizer.transform(data_baru_processed['Lemmatized'].apply(' '.join))
            st.write("TF-IDF Shape:", sinopsis_tfidf.shape)
        
            prediksi = loaded_model.predict(sinopsis_tfidf.toarray())
            st.success(f"Genre yang diprediksi: **{prediksi[0]}**")
        except Exception as e:
            st.write("Checkpoint: Preprocessor instance dibuat")


