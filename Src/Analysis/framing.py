import pandas as pd
from collections import Counter, defaultdict
import unicodedata
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import matplotlib
matplotlib.use("Agg")  # ou "TkAgg" si tu veux l'affichage interactif


# Charger le modèle SpaCy français
nlp = spacy.load("fr_core_news_sm")

# Chargement des données
articles = pd.read_json("data/Processed/polarite_cameroun_rss.json")

# Chargement du lexique
lexique = pd.read_csv("data/lexique_emotionnel.csv", encoding="utf-8")
lexique['mot'] = lexique['mot'].str.lower()
lexique['polarité'] = lexique['polarité'].str.lower()
lexique['émotion'] = lexique['émotion'].str.lower()

# Dictionnaires de correspondance
pol_dict = dict(zip(lexique['mot'], lexique['polarité']))
emo_dict = dict(zip(lexique['mot'], lexique['émotion']))

# Fonction d’analyse émotionnelle
def analyse_emotionnelle(tokens):
    if not isinstance(tokens, list): return pd.Series([0, 0, 0, 0])
    tokens = [t.lower() for t in tokens]
    polarités = [pol_dict.get(t) for t in tokens]
    
    total = len(tokens)
    n_pos = polarités.count("positive")
    n_neg = polarités.count("negative")
    n_neutre = polarités.count("neutral")
    n_emo = n_pos + n_neg
    
    return pd.Series([
        n_emo / total if total else 0,
        n_pos / total if total else 0,
        n_neg / total if total else 0,
        n_neutre / total if total else 0,
    ])

# Calcul des proportions
articles[[
    "proportion_émotion",
    "proportion_positive",
    "proportion_négative",
    "proportion_neutre"
]] = articles['tokens_Texte'].apply(analyse_emotionnelle)

# Détection des polarités et émotions
def detect_polarities(tokens):
    if not isinstance(tokens, list): return []
    return [pol_dict.get(t.lower()) for t in tokens if pol_dict.get(t.lower())]

def detect_emotions(tokens):
    if not isinstance(tokens, list): return []
    return [emo_dict.get(t.lower()) for t in tokens if emo_dict.get(t.lower()) not in [None, "neutre"]]

articles["polarités"] = articles["tokens_Texte"].apply(detect_polarities)
articles["émotions"] = articles["tokens_Texte"].apply(detect_emotions)

# Lemmatisation
def lemmatise_tokens(tokens):
    if not isinstance(tokens, list): return []
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

articles["lemmas_Texte"] = articles["tokens_Texte"].apply(lemmatise_tokens)

# Date et regroupement par mois
articles["Date"] = pd.to_datetime(articles["Date"], errors="coerce")
articles["Mois"] = articles["Date"].dt.to_period("M").astype(str)

# Label sentiment majoritaire
def sentiment_majoritaire(pols):
    if not pols: return "neutre"
    return Counter(pols).most_common(1)[0][0]

articles["label_sentiment"] = articles["polarités"].apply(sentiment_majoritaire)

# ========================
# VISUALISATIONS
# ========================

# 1. Répartition relative des émotions par média
emotion_counts = defaultdict(Counter)
for _, row in articles.iterrows():
    media = row["Nom du média"]
    for emo in row["émotions"]:
        emotion_counts[media][emo] += 1

emotion_df = pd.DataFrame(emotion_counts).fillna(0).T.astype(int)
emotion_df_norm = emotion_df.div(emotion_df.sum(axis=1), axis=0)

emotion_df_norm.plot(kind="bar", stacked=True, figsize=(14, 6), colormap="Set2")
plt.title("Répartition relative des émotions par média (normalisé)")
plt.ylabel("Proportion d’émotions")
plt.xlabel("Nom du média")
plt.xticks(rotation=45)
plt.legend(title="Émotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 2. Évolution temporelle des émotions
emo_time = defaultdict(Counter)
for _, row in articles.iterrows():
    mois = row["Mois"]
    for emo in row["émotions"]:
        emo_time[mois][emo] += 1

emo_time_df = pd.DataFrame(emo_time).fillna(0).T.sort_index().astype(int)

emo_time_df.plot(figsize=(14, 6), marker='o')
plt.title("Évolution des émotions au fil du temps")
plt.xlabel("Mois")
plt.ylabel("Nombre d’occurrences")
plt.xticks(rotation=45)
plt.legend(title="Émotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 3. Répartition des émotions par polarité
emotion_polarity_counts = Counter()
for _, row in articles.iterrows():
    for emo, pol in product(row["émotions"], row["polarités"]):
        emotion_polarity_counts[(emo, pol)] += 1

cross_df = pd.DataFrame.from_dict(emotion_polarity_counts, orient="index", columns=["count"])
cross_df.index = pd.MultiIndex.from_tuples(cross_df.index, names=["émotion", "polarité"])
cross_df = cross_df.reset_index()
pivot = cross_df.pivot(index="émotion", columns="polarité", values="count").fillna(0).astype(int)

pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab10")
plt.title("Répartition des émotions par polarité")
plt.xlabel("Émotion")
plt.ylabel("Nombre d’occurrences")
plt.xticks(rotation=45)
plt.legend(title="Polarité")
plt.tight_layout()
plt.show()

# 4. Intensité émotionnelle par média
plt.figure(figsize=(12, 6))
sns.boxplot(data=articles, x="Nom du média", y="proportion_émotion")
plt.xticks(rotation=45)
plt.title("Intensité émotionnelle des articles par média")
plt.tight_layout()
plt.show()


# ANALYSE DU BIAIS DE SOURCE

# 1. Polarité moyenne par média
polarite_moyenne = articles.groupby("Nom du média")["polarite"].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=polarite_moyenne.values, y=polarite_moyenne.index, hue=polarite_moyenne.index, palette="coolwarm", legend=False)
plt.title("Polarité moyenne par média")
plt.xlabel("Polarité moyenne")
plt.ylabel("Nom du média")
plt.tight_layout()
plt.show()

# 2. Distribution brute des sentiments
distribution = articles.groupby(["Nom du média", "label_sentiment"]).size().unstack(fill_value=0)

distribution.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
plt.title("Distribution brute des sentiments par média")
plt.ylabel("Nombre d'articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Distribution proportionnelle
distribution_norm = distribution.div(distribution.sum(axis=1), axis=0)

distribution_norm.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
plt.title("Distribution proportionnelle des sentiments par média")
plt.ylabel("Proportion d'articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Score de biais (écart absolu entre positifs et négatifs)
if "positive" in distribution_norm.columns and "negative" in distribution_norm.columns:
    score_biais = (distribution_norm["positive"] - distribution_norm["negative"]).abs()

    score_biais.sort_values().plot(kind="barh", color="darkred", figsize=(10, 6))
    plt.title("Score de biais de polarité par média (|positive - negative|)")
    plt.xlabel("Écart de polarité")
    plt.tight_layout()
    plt.show()
else:
    print("Les colonnes 'positive' et 'negative' sont absentes. Vérifiez les labels dans 'label_sentiment'.")
