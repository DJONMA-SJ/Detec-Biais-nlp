import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# Charger les fichiers pré-traités
fichiers = [
    'data/Processed/actudata_cleaned.csv',
    'data/Processed/camdata_cleaned.csv',
    'data/Processed/presidata_cleaned.csv',
    'data/Processed/rssdata_cleaned.csv'
]

# Initialiser VADER
analyzer = SentimentIntensityAnalyzer()

def analyse_sentiment_vader(text):
    try:
        # Traduction du texte en anglais
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        score = analyzer.polarity_scores(translated)
        return score
    except Exception as e:
        print(f"Erreur de traitement : {e}")
        return {}

for fichier in fichiers:
    print(f"Analyse de {fichier}...")
    df = pd.read_csv(fichier)
    
    # Appliquer l'analyse VADER
    df['vader_sentiment'] = df['cleaned_Texte intégral'].astype(str).apply(analyse_sentiment_vader)

    # Sauvegarder au format JSON pour meilleure visibilité
    json_output = fichier.replace('csv', 'vader.json')
    df.to_json(json_output, orient='records', force_ascii=False, indent=2)
    print(f"Résultats sauvegardés dans {json_output}")
