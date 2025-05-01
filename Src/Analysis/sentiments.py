import pandas as pd
import json
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# === CHEMINS ===
csv_input_path = "C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Raw/Presidentielle_2025.csv"
csv_output_csv = "C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Analyses/Presi_sentiments.csv"
json_output_path = "C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Analyses/Presi_sentiments.json"
model_path = "models/naive_bayes_model3.pkl"

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv(csv_input_path)
text_column = "Texte intégral"

# === EMBEDDINGS AVEC UN MODELE LEGER
print("[*] Génération des embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = df[text_column].fillna("").tolist()
embeddings = model.encode(texts)

# === ANNOTATION MANUELLE POUR ENTRAINEMENT (EXEMPLE) ===
import random
random.seed(42)

n_samples = len(embeddings)
possible_labels = ["positive", "negative", "neutral"]
labels = [random.choice(possible_labels) for _ in range(n_samples)]


# === ENTRAINEMENT DU MODELE
print("[*] Entraînement du modèle Naive Bayes...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)

accuracy = model_nb.score(X_test_scaled, y_test)
print(f"[✓] Accuracy du modèle : {accuracy:.2f}")

# === SAUVEGARDE DU MODELE
joblib.dump(model_nb, model_path)

# === PREDICTION SUR TOUT LE CORPUS
print("[*] Prédiction des sentiments...")
all_embeddings_scaled = scaler.transform(embeddings)
predicted_sentiments = model_nb.predict(all_embeddings_scaled)

# === SAUVEGARDE DES RESULTATS
df["naive_sentiment"] = predicted_sentiments
df.to_csv(csv_output_csv, index=False, encoding="utf-8-sig")

with open(json_output_path, "w", encoding="utf-8") as f_json:
    json.dump(df.to_dict(orient="records"), f_json, indent=2, ensure_ascii=False)

print(f"[✓] Résultats sauvegardés en CSV : {csv_output_csv}")
print(f"[✓] Résultats sauvegardés en JSON : {json_output_path}")
