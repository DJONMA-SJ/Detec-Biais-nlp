import feedparser
import csv

# URL du flux RSS pour le Cameroun en français
rss_url = 'https://news.google.com/rss/search?q=presidentielle_2025+cameroon&hl=fr&gl=CM&ceid=CM:fr'

# Parser le flux RSS
feed = feedparser.parse(rss_url)

# Structure CSV
csv_filename = "articles_cameroun_rss.csv"
fieldnames = ['title', 'content', 'published', 'source', 'url', 'theme']

# Écriture dans le fichier CSV
with open(csv_filename, mode='w', encoding='utf-8', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Extraire les articles du flux RSS
    for entry in feed.entries:
        # Extraction des données
        title = entry.get('title', 'N/A')
        content = entry.get('summary', 'N/A')  # ou entry.get('content', 'N/A') pour plus de détails
        published = entry.get('published', 'N/A')
        source = entry.get('source', {}).get('title', 'N/A')  # Si disponible
        url = entry.get('link', 'N/A')

        # Pour le thème, on peut ajouter une logique ici, par exemple par mot-clé dans le titre
        theme = 'politique'  # Ajuste si tu veux automatiser selon le contenu du titre

        # Écriture de l'article dans le fichier CSV
        writer.writerow({
            'title': title,
            'content': content,
            'published': published,
            'source': source,
            'url': url,
            'theme': theme
        })

print(f"{len(feed.entries)} articles enregistrés dans '{csv_filename}'")
