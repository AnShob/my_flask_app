import streamlit as st
import pandas as pd
from joblib import load
import nltk
from preprosesing import Preprocessor

# Pastikan dataset NLTK terunduh
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Memuat model dan vectorizer
loaded_model = load('svm_model(RBF).joblib')
loaded_vectorizer = load('tfidf_vectorizer.joblib')

# Instance preprocessor
preprocessor = Preprocessor(sinopsis_column='Sinopsis')

st.title("Klasifikasi Genre Buku")
st.write("Masukkan sinopsis buku untuk memprediksi genre.")

sinopsis = st.text_area("Masukkan Sinopsis:")

if st.button("Prediksi Genre"):
    if sinopsis.strip():
        data_baru = pd.DataFrame({'Sinopsis': [sinopsis]})
        data_baru_processed = preprocessor.preprocess_dataframe(data_baru)

        # Transformasikan ke TF-IDF
        sinopsis_tfidf = loaded_vectorizer.transform(data_baru_processed['Lemmatized'].apply(' '.join))
        prediksi = loaded_model.predict(sinopsis_tfidf.toarray())

        st.success(f"Genre yang diprediksi: **{prediksi[0]}**")
    else:
        st.error("Silakan masukkan sinopsis.")
