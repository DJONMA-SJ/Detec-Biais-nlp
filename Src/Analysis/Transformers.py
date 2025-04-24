from transformers import pipeline
import pandas as pd

# Chargement des données prétraitées
df = pd.read_csv("data/Processed/rssdata_cleaned.csv")  # adapte ici selon le fichier

# Initialisation du pipeline HuggingFace
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Application du modèle (sur les 50 premières lignes pour tester)
df['sentiment'] = df['cleaned_Texte intégral'].astype(str).apply(lambda x: classifier(x[:512])[0]['label'])  # tronque à 512 tokens

# Affichage des résultats
print(df[['Titre', 'sentiment']].head())

# Export en JSON
df[['Titre', 'cleaned_Texte intégral', 'sentiment']].to_json("data/Analyses/transformers_sentiment.json", orient="records", force_ascii=False, indent=2)
