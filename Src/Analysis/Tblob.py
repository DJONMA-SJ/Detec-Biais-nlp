from textblob import TextBlob
import pandas as pd

# Charger les données prétraitées
df = pd.read_csv("data/Processed/presidata_cleaned.csv")

# Appliquer TextBlob
df["tb_polarity"] = df["cleaned_Texte intégral"].apply(lambda text: TextBlob(text).sentiment.polarity)
df["tb_subjectivity"] = df["cleaned_Texte intégral"].apply(lambda text: TextBlob(text).sentiment.subjectivity)

# Aperçu
print(df[["cleaned_Texte intégral", "tb_polarity", "tb_subjectivity"]].head())

# Exporter si besoin
df.to_json("data/Analyses/textblob_sentiment.json", orient="records", force_ascii=False, indent=2)
