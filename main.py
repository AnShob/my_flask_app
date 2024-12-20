# from flask import Flask, render_template, request
# from datetime import datetime
# import pickle
# from preprosesing import Preprocessor
# import pandas as pd
# import os


# app = Flask(__name__)

# with open('svm_model(RBF).pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#     loaded_vectorizer = pickle.load(vectorizer_file)

# preprocessor = Preprocessor(sinopsis_column='Sinopsis')


# @app.route('/', methods=['GET', 'POST'])
# def main():
#     if request.method == 'POST':
#         # Ambil data dari form
#         sinopsis = request.form['Sinopsis']
#         # Membuat DataFrame untuk data baru
#         data_baru = pd.DataFrame({'Sinopsis': [sinopsis]})
#         # Preprocessing data baru
#         data_baru_processed = preprocessor.preprocess_dataframe(data_baru)
        
#         # Prediksi dengan model
#         sinopsis_tfidf = loaded_vectorizer.transform(data_baru_processed['Lemmatized'].apply(' '.join))
#         sinopsis_dense = sinopsis_tfidf.toarray()  # Ubah sparse matrix menjadi dense array
#         prediksi = loaded_model.predict(sinopsis_dense)
        
        
#         # Tampilkan genre yang diprediksi di halaman
#         genre_prediksi = prediksi[0]  # Ambil genre pertama dari hasil prediksi
#         return render_template('cek_usia.html', Sinopsis=sinopsis, Genre=genre_prediksi)
#     return render_template('cek_usia.html',  Sinopsis=None, Genre=None, Foto=None)
    
# if __name__ == "__main__":
#     app.run(host='0.0.0.0')

from flask import Flask, render_template, request
from datetime import datetime
import pickle
from preprosesing import Preprocessor
import pandas as pd
import os
import requests

app = Flask(__name__)

# Google Drive file IDs
MODEL_FILE_URL = "https://drive.google.com/uc?id=xyUxJAoHSsflkGjGQUqPGwF9cVgvkwj"
# VECTORIZER_FILE_URL = "https://drive.google.com/uc?id=<file_id_vectorizer>"

# File names
MODEL_FILE_NAME = 'svm_model(RBF).pkl'
# VECTORIZER_FILE_NAME = 'tfidf_vectorizer.pkl'


def download_file(url, file_name):
    """Download file from URL if not present locally."""
    if not os.path.exists(file_name):
        print(f"Downloading {file_name} from Google Drive...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"{file_name} downloaded successfully.")
        else:
            raise Exception(f"Failed to download {file_name}. Status code: {response.status_code}")


# Download the required files
try:
    download_file(MODEL_FILE_URL, MODEL_FILE_NAME)
    # download_file(VECTORIZER_FILE_URL, VECTORIZER_FILE_NAME)
except Exception as e:
    print(f"Error during file download: {e}")
    exit(1)

# Load the model and vectorizer
with open(MODEL_FILE_NAME, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Preprocessor instance
preprocessor = Preprocessor(sinopsis_column='Sinopsis')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Ambil data dari form
        sinopsis = request.form['Sinopsis']
        # Membuat DataFrame untuk data baru
        data_baru = pd.DataFrame({'Sinopsis': [sinopsis]})
        # Preprocessing data baru
        data_baru_processed = preprocessor.preprocess_dataframe(data_baru)
        
        # Prediksi dengan model
        sinopsis_tfidf = loaded_vectorizer.transform(data_baru_processed['Lemmatized'].apply(' '.join))
        sinopsis_dense = sinopsis_tfidf.toarray()  # Ubah sparse matrix menjadi dense array
        prediksi = loaded_model.predict(sinopsis_dense)
        
        # Tampilkan genre yang diprediksi di halaman
        genre_prediksi = prediksi[0]  # Ambil genre pertama dari hasil prediksi
        return render_template('cek_usia.html', Sinopsis=sinopsis, Genre=genre_prediksi)
    return render_template('cek_usia.html', Sinopsis=None, Genre=None, Foto=None)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
