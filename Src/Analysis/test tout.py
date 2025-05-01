import pandas as pd

# Adapter le chemin à votre fichier
df = pd.read_json("data/Analyses/rss_sentiment.json") 
print(df.columns)


texts = df['Texte intégral'].tolist()
labels_text = df['naive_sentiment'].tolist()
  
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
labels = [label_mapping[label.lower()] for label in labels_text]


from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

embeddings = []

for text in tqdm(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[0][0]  # [CLS] token
    embeddings.append(cls_embedding.numpy())

import numpy as np

embeddings = np.array(embeddings)
labels = np.array(labels)

np.save("embeddings2.npy", embeddings)
np.save("labels2.npy", labels)

print("Fichiers enregistrés : embeddings2.npy et labels.npy")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Exemple : charger vos embeddings et labels
X = np.load("data/Analyses/embeddings2.npy")         # Embeddings BERT
y = np.load("data/Analyses/labels2.npy")             # Labels (0: négatif, 1: neutre, 2: positif)

# Séparer les données en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = GaussianNB()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=["Négatif", "Neutre", "Positif"]))



## pour le fichier polarite_json
import json
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

# Charger le fichier JSON
with open("data/Processed/articles_cameroun_rss.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Fonction pour calculer la polarité
def get_sentiment(text):
    blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    return blob.sentiment[0]

# Fonction pour générer un label à partir de la polarité
def get_label(score):
    if score > 0.1:
        return "positif"
    elif score < -0.1:
        return "négatif"
    else:
        return "neutre"

# Appliquer la polarité à chaque entrée
for article in data:
    texte = article.get("cleaned_Texte intégral", "")
    polarite = get_sentiment(texte)
    article["polarite"] = polarite
    article["label_sentiment"] = get_label(polarite)

# Sauvegarder les résultats dans un nouveau fichier JSON
with open("data/Processed/polarite_cameroun_rss.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
