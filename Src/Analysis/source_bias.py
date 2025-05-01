import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier JSON
with open("data/Processed/polarite_cameroun_rss.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Créer le DataFrame
df = pd.DataFrame(data)

# 1. Polarité moyenne par média

polarite_moyenne = df.groupby("Nom du média")["polarite"].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=polarite_moyenne.values, y=polarite_moyenne.index, palette="coolwarm")
plt.title("Polarité moyenne par média")
plt.xlabel("Polarité moyenne")
plt.ylabel("Nom du média")
plt.tight_layout()
plt.show()

# 2. Distribution brute des sentiments par média
distribution = df.groupby(["Nom du média", "label_sentiment"]).size().unstack(fill_value=0)

distribution.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
plt.title("Distribution brute des sentiments par média")
plt.ylabel("Nombre d'articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Distribution normalisée (proportionnelle)
distribution_norm = distribution.div(distribution.sum(axis=1), axis=0)

distribution_norm.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
plt.title("Distribution proportionnelle des sentiments par média")
plt.ylabel("Proportion d'articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Score de biais de polarité (écart absolu)
score_biais = (distribution_norm["positive"] - distribution_norm["negative"]).abs()

score_biais.sort_values().plot(kind="barh", color="darkred", figsize=(10, 6))
plt.title("Score de biais de polarité par média (|positive - negative|)")
plt.xlabel("Écart de polarité")
plt.tight_layout()
plt.show()

# 5. Polarité par média et par thème
pivot_theme = df.groupby(["Thème", "Nom du média"])["polarite"].mean().unstack()

pivot_theme.plot(kind="bar", figsize=(15, 6), colormap="viridis")
plt.title("Polarité moyenne par média selon les thèmes")
plt.ylabel("Polarité moyenne")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
