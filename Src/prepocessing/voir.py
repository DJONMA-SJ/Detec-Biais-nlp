import pandas as pd

# Charger le fichier CSV
file_path = 'C:/Users/GENIUS ELECTRONICS/Desktop/Detec-Biais-nlp/data/Raw/articles_cameroun_rss.csv'  # Remplace par ton chemin de fichier
df = pd.read_csv(file_path)

# Afficher les premières lignes pour inspection
print("Les premières lignes des données:")
print(df.head())

# Afficher les colonnes pour voir la structure
print("/nLes colonnes du fichier:")
print(df.columns)
