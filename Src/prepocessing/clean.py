import pandas as pd
import nltk
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.fr.stop_words import STOP_WORDS
import string

# Charger le modèle SpaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Charger le fichier CSV
file_path = 'C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Raw/Presidentielle_2025.csv'  # Remplace par ton chemin de fichier
df = pd.read_csv(file_path)

# Fonction pour nettoyer et prétraiter le texte
def clean_text(text):
    # Enlever les caractères spéciaux et les chiffres
    text = re.sub(r'/W', ' ', str(text))

    # Convertir en minuscule
    text = text.lower()
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les mots vides
    stop_words = set(STOP_WORDS)
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenisation avec SpaCy
    doc = nlp(text)
    
    # Lemmatisation et suppression des stopwords
    lemmatized_text = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
    # Rejoindre les tokens en une seule chaîne
    return " ".join(lemmatized_text)

# Appliquer le nettoyage aux colonnes 'Titre' et 'Texte intégral'
df['cleaned_Titre'] = df['Titre'].apply(clean_text)
df['cleaned_Texte intégral'] = df['Texte intégral'].apply(clean_text)

# Afficher les premières lignes des données nettoyées
print("Premières lignes après nettoyage:")
print(df[['Titre', 'cleaned_Titre', 'Texte intégral', 'cleaned_Texte intégral']].head())

# Sauvegarder les données nettoyées dans un nouveau fichier CSV
df.to_csv('presidata_cleaned.csv', index=False)
