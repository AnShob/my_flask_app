import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, sinopsis_column='Sinopsis'):
        self.sinopsis_column = sinopsis_column
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def lowercasing(self, df):
        df[self.sinopsis_column] = df[self.sinopsis_column].str.lower()
        return df
    
    def clean_text(self, text): # Remove Punctuation
        text = re.sub(r'[^\w\s.]', '', text)  # Hapus karakter non-alfanumerik
        text = re.sub(r'\.{2,}', '.', text)   # Hapus tanda titik berlebihan
        text = text.strip()                  # Hilangkan spasi di awal/akhir
        return text
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_dataframe(self, df):
        # Pastikan kolom sinopsis tersedia
        if self.sinopsis_column not in df.columns:
            raise ValueError(f"Kolom '{self.sinopsis_column}' tidak ditemukan dalam dataset")
        
        # Drop baris kosong
        df = df.dropna(subset=[self.sinopsis_column])

        # Proses preprocessing
        df[self.sinopsis_column] = df[self.sinopsis_column].apply(self.clean_text)
        df['Tokenized'] = df[self.sinopsis_column].apply(self.tokenize)
        df['Without_Stopwords'] = df['Tokenized'].apply(self.remove_stopwords)
        df['Lemmatized'] = df['Without_Stopwords'].apply(self.lemmatize)
    
        return df