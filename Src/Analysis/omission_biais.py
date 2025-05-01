import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Charger les articles
articles = pd.read_json("C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Processed/polarite_cameroun_rss.json")

# Charger le lexique
lexique = pd.read_csv("C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/lexique_emotionnel.csv", encoding="utf-8")
lexique['mot'] = lexique['mot'].str.lower()

# Dictionnaires pour recherche rapide
pol_dict = dict(zip(lexique['mot'], lexique['polarité']))
emo_dict = dict(zip(lexique['mot'], lexique['émotion']))

# Regrouper tous les tokens par média (indépendamment du thème ici)
tokens_par_media = defaultdict(list)

for _, row in articles.iterrows():
    media = row.get("Nom du média", "Inconnu")
    tokens = row.get("tokens_Texte", [])
    if isinstance(tokens, str):
        try:
            tokens = eval(tokens)
        except:
            tokens = []
    tokens = [t.lower() for t in tokens if isinstance(t, str) and len(t) > 2]
    tokens_par_media[media].extend(tokens)
# Compter tous les tokens pour trouver les plus fréquents
global_freq = Counter()

for tokens in tokens_par_media.values():
    global_freq.update(tokens)

# Choisir les 30 tokens les plus fréquents (modifiable)
top_tokens = [t for t, _ in global_freq.most_common(30)]


# Initialiser DataFrame
matrix = pd.DataFrame(index=tokens_par_media.keys(), columns=top_tokens).fillna(0)

# Remplir les fréquences
for media, tokens in tokens_par_media.items():
    token_counts = Counter(tokens)
    for token in top_tokens:
        matrix.loc[media, token] = token_counts.get(token, 0)

# Nettoyer les tokens et reformer le corpus
def clean_tokens(x):
    if isinstance(x, str):
        try:
            tokens = eval(x)
        except:
            return ""
    elif isinstance(x, list):
        tokens = x
    else:
        return ""
    
    tokens = [t for t in tokens if isinstance(t, str) and len(t) > 2]
    return " ".join(tokens)

corpus_textes = articles["tokens_Texte"].apply(clean_tokens)

# Optionnel : retirer les lignes vides
corpus_textes = corpus_textes[corpus_textes.str.strip().astype(bool)]

vectorizer = CountVectorizer(max_df=0.95, min_df=2)  # pas de stop_words
X = vectorizer.fit_transform(corpus_textes)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
def afficher_themes(model, feature_names, n_top_words=10):
    for idx, topic in enumerate(model.components_):
        print(f"\n Thème {idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

afficher_themes(lda, vectorizer.get_feature_names_out())
articles["topic_lda"] = lda.transform(X).argmax(axis=1)

def afficher_themes(model, feature_names, n_top_words=10):
    for idx, topic in enumerate(model.components_):
        print(f"\nThème {idx} :")
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print("Mots-clés :", ", ".join(top_words))

afficher_themes(lda, vectorizer.get_feature_names_out())
theme_labels = {
    0: "Opposition & démocratie",
    1: "Élections locales",
    2: "Gouvernement & administration",
    3: "Église et société",
    4: "Campagne présidentielle"
}

# Appliquer le label dans le DataFrame
articles["theme_label"] = articles["topic_lda"].map(theme_labels)
# Compter combien d’articles chaque média consacre à chaque thème
table = pd.crosstab(articles["Nom du média"], articles["topic_lda"])
# Médias qui n'ont rien publié sur un thème
omissions = {}

for media in table.index:
    themes_omits = table.columns[table.loc[media] == 0].tolist()
    if themes_omits:
        omissions[media] = themes_omits
print("Médias n'ayant couvert aucun article sur certains thèmes :")

for media, themes in omissions.items():
    noms_themes = [theme_labels.get(t, f"Thème {t}") for t in themes]
    print(f"\n {media} omet : {', '.join(noms_themes)}")

# Matrice binaire : 1 = le média a au moins 1 article sur ce thème, 0 = omission
binary_coverage = table.copy()
binary_coverage[binary_coverage > 0] = 1
# Bien reformuler les noms de colonnes
binary_coverage.columns = [theme_labels.get(int(i), f"Thème {int(i)}") for i in binary_coverage.columns]



# Appliquer log(1 + x) pour adoucir les grandes valeurs
matrix_log = matrix.applymap(lambda x: np.log1p(x))

plt.figure(figsize=(15, 8))
sns.heatmap(matrix_log, cmap="mako", annot=False, linewidths=0.5)

plt.title("Fréquence (log) des tokens les plus fréquents par média")
plt.xlabel("Tokens")
plt.ylabel("Médias")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=articles, x="topic_lda", hue="Nom du média", palette="Set2")
plt.title("Répartition des thèmes LDA par média (avec scikit-learn)")
plt.xlabel("Thème latent")
plt.ylabel("Nombre d'articles")
plt.xticks(rotation=0)
plt.legend(title="Média", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(binary_coverage, cmap="Greens", linewidths=0.5, annot=True, cbar=False)

plt.title("Thèmes LDA couverts (1) ou omis (0) par média")
plt.xlabel("Thèmes")
plt.ylabel("Médias")
plt.xticks(rotation=20, ha="right")  # inclinaison propre
plt.tight_layout()
plt.show()
