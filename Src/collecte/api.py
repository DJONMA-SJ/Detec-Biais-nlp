import requests
import csv

API_KEY = 'c22e5e421f8e0d2011ad165bb61065a5'  # la clé API
URL = 'https://gnews.io/api/v4/search'

# Paramètres de la requête
params = {
   'q': 'politique OR élections',
    'lang': 'fr',
    'country': 'cm',
    'max': 5000,
    'token': API_KEY
}

# Requête à l’API
response = requests.get(URL, params=params)

# Debug si besoin
print("Status:", response.status_code)
print("Response preview:", response.text[:300])  # Affiche les 300 premiers caractères

try:
    data = response.json()
except Exception as e:
    print("Erreur lors du décodage JSON :", e)
    data = {}

# Structure CSV
csv_filename = "articles_cameroun.csv"
fieldnames = ['title', 'content', 'publishedAt', 'source', 'url', 'theme']

# Écriture dans le fichier CSV
with open(csv_filename, mode='w', encoding='utf-8', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for article in data.get('articles', []):
        writer.writerow({
            'title': article.get('title'),
            'content': article.get('content'),
            'publishedAt': article.get('publishedAt'),
            'source': article.get('source', {}).get('name'),
            'url': article.get('url'),
            'theme': 'politique'
        })

print(f"{len(data.get('articles', []))} articles enregistrés dans '{csv_filename}'")
