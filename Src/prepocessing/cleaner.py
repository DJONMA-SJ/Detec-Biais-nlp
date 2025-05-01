import os
import pandas as pd
import spacy
from bs4 import BeautifulSoup
import re
import json
from nltk.corpus import stopwords
import nltk

# === Initialisation ===
nltk.download('stopwords')
stop_words = set(stopwords.words("french"))
nlp = spacy.load("fr_core_news_sm")

# === CHEMINS ===
RAW_DIR = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Raw"
OUTPUT_DIR = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Processed"

# === CRÉER LE DOSSIER DE SORTIE SI INEXISTANT ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FONCTION DE NETTOYAGE + TOKENISATION + LEMMATISATION ===
def preprocess_text(text):
    if pd.isna(text):
        return "", []
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = text.lower()

    doc = nlp(text)
    tokens = [token.text for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
    lemmas = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]

    return " ".join(lemmas), tokens

# === TRAITEMENT PAR FICHIER CSV ===
for filename in os.listdir(RAW_DIR):
    if filename.endswith(".csv"):
        input_path = os.path.join(RAW_DIR, filename)
        print(f"[*] Traitement du fichier : {filename}")

        df = pd.read_csv(input_path)

        # Prétraitement du texte intégral
        cleaned_texts = []
        tokens_list = []

        for text in df["Texte intégral"].astype(str):
            cleaned, tokens = preprocess_text(text)
            cleaned_texts.append(cleaned)
            tokens_list.append(tokens)

        df["cleaned_Texte intégral"] = cleaned_texts
        df["tokens_Texte"] = tokens_list

        # Idem pour les titres
        df["cleaned_Titre"] = df["Titre"].astype(str).apply(lambda t: preprocess_text(t)[0])

        # Export en JSON
        articles = df.to_dict(orient="records")
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", ".json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"Exporté vers : {output_path}")

