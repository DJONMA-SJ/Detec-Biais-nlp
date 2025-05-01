from Src.Preprocessing.cleaner import preprocess_articles
from Src.Analysis.sentiments import analyze_sentiment
from Src.Analysis.omission_biais import extract_named_entities
from Src.Analysis.framing import detect_framing_bias
from Src.Analysis.source_bias import analyze_source_bias
from Src.Analysis.lexicon_bias import detect_lexical_bias
from Src.Visualization.dashboards import generate_dashboard

# === Étape 1 : Prétraitement ===
articles = preprocess_articles("data/Raw/")

# === Étape 2 : Analyse des sentiments ===
articles = analyze_sentiment(articles)

# === Étape 3 : Extraction d'entités nommées ===
articles = extract_named_entities(articles)

# === Étape 4 : Détection du cadrage ===
articles = detect_framing_bias(articles)

# === Étape 5 : Analyse du biais de source ===
articles = analyze_source_bias(articles)

# === Étape 6 : Analyse lexicale (subjectivité) ===
articles = detect_lexical_bias(articles)

# === Étape 7 : Visualisation / Export ===
generate_dashboard(articles)


import os
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler

# === Initialisation du modèle BERT ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # Mode évaluation

# === CHEMINS ===
INPUT_DIR = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Processed"
OUTPUT_DIR = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Fonction d'extraction d'embedding [CLS] ===
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# === Parcours des fichiers prétraités ===
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        print(f"[*] Traitement de : {filename}")
        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            articles = json.load(f)

        embeddings_list = []
        texts_cleaned = []

        for article in articles:
            text = article.get("cleaned_Texte intégral", "")
            if not text.strip():
                continue
            emb = get_embedding(text)
            embeddings_list.append(emb.tolist())
            texts_cleaned.append(text)

        # Normalisation
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_list)

        # Export en JSON
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".json", "_embeddings.json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([
                {"text": txt, "embedding": emb.tolist()}
                for txt, emb in zip(texts_cleaned, embeddings_scaled)
            ], f, ensure_ascii=False, indent=2)

        print(f"Embeddings exportés : {output_path}")


