import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === CHEMINS ===
csv_input_path = r"C:\Users\GENIUS ELECTRONICS\Desktop\Detec-Biais-nlp\data\Raw\articles actucameroun.csv"
csv_output_path = r"C:\Users\GENIUS ELECTRONICS\Desktop\Detec-Biais-nlp\data\Raw\articles_labeled.csv"

# === ANALYSEUR VADER ===
analyzer = SentimentIntensityAnalyzer()

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv(csv_input_path)
df = df.dropna(subset=["Texte intégral"])

# === FONCTION DE LABELISATION ===
def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# === APPLICATION ===
df["label"] = df["Texte intégral"].apply(get_sentiment_label)

# === SAUVEGARDE ===
df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
print(f"[✓] Fichier avec labels sauvegardé : {csv_output_path}")


import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# === CHEMINS ===
csv_input_path = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Raw/articles_labeled3.csv"
model_output_path = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/models/naive_bayes_model3.pkl"

# === CHARGEMENT BERT ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv(csv_input_path)
df = df.dropna(subset=["Texte intégral", "label"])

# === EMBEDDINGS BERT ===
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# === CONSTRUCTION X, y ===
X = []
y = []

print("[*] Génération des embeddings...")

for _, row in df.iterrows():
    try:
        emb = get_bert_embedding(row["Texte intégral"])
        X.append(emb)
        y.append(row["label"])
    except Exception as e:
        print(f"Erreur texte ignoré: {e}")

# === NORMALISATION ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === ENTRAÎNEMENT ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)

# === ÉVALUATION ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" Accuracy du modèle : {acc:.2f}")

# === SAUVEGARDE ===
joblib.dump(model, model_output_path)
print(f" Modèle sauvegardé : {model_output_path}")


import pandas as pd
import torch
import joblib
import json
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler

# === CHEMINS ===
csv_input_path = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Raw/Presidentielle_2025.csv"
json_output_path = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Analyses/Presidensielle_sentiments.json"
model_path = r"C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/models/naive_bayes_model.pkl"

# === CHARGEMENT DU MODELE ET TOKENIZER ===
naive_bayes_model = joblib.load(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv(csv_input_path)

# === FONCTION POUR OBTENIR LES EMBEDDINGS BERT ===
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# === ANALYSE DES TEXTES ===
results = []
scaler = StandardScaler()

for _, row in df.iterrows():
    texte = str(row["Texte intégral"])
    try:
        if not texte.strip():
            sentiment = "neutral"
        else:
            emb = get_bert_embedding(texte).reshape(1, -1)
            emb_scaled = scaler.fit_transform(emb)
            sentiment = naive_bayes_model.predict(emb_scaled)[0]
    except Exception as e:
        sentiment = "error"

    result = {
        "Titre": row["Titre"],
        "Texte intégral": row["Texte intégral"],
        "Date": row["Date"],
        "Nom du média": row["Nom du média"],
        "URL": row["URL"],
        "Thème": row["Thème"],
        "naive_sentiment": sentiment
    }
    results.append(result)

# === SAUVEGARDE EN JSON ===
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[✓] Fichier JSON enrichi sauvegardé : {json_output_path}")
