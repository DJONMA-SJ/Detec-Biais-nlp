import cloudscraper
from bs4 import BeautifulSoup
import time
import random
import csv

# Liste des URLs d'articles
urls = [
    "https://www.cameroon-tribune.cm/article.html/69972/fr.html/lutte-contre-les-changements-climatiques-3-5-milliards-de-f",
    "https://www.cameroon-tribune.cm/article.html/69970/fr.html/presidentielle-2025-les-deputes-apotres-de-la-paix",
    "https://www.cameroon-tribune.cm/article.html/69891/fr.html/presidentielle-2025-paul-biya-peut-compter-sur-la",
    "https://www.cameroon-tribune.cm/article.html/69646/fr.html/presidentielle-2025-les-chefs-du-nkam-appellent-la-candidature-de",
    "https://www.cameroon-tribune.cm/article.html/69299/fr.html/presidentielle-2025-joshua-osih-candidat-du",
    "https://www.cameroon-tribune.cm/article.html/69201/fr.html/presidentielle-2025-le-nde-choisit-paul-biya",
    "https://www.cameroon-tribune.cm/article.html/68972/fr.html/presidentielle-2025-les-leaders-musulmans-derriere-paul",
    "https://www.cameroon-tribune.cm/article.html/69970/fr.html/presidentielle-2025-les-deputes-apotres-de-la-paix",
    "https://www.cameroon-tribune.cm/article.html/69969/fr.html/upcoming-presidential-election-senate-calls-for",
    "https://www.cameroon-tribune.cm/article.html/69968/fr.html/securisation-du-transport-marchandises-dangereuses-le-cameroun-pare"
]

# Simuler un navigateur pour éviter le blocage
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# Créer un scraper compatible Cloudflare
scraper = cloudscraper.create_scraper()

# Créer le fichier CSV
with open("articles_cameroontribune.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Titre", "Texte intégral", "Date de publication", "Source", "URL", "Thème"])

    # Parcours de chaque article
    for url in urls:
        try:
            response = scraper.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")

                # Titre
                titre = soup.find("h1").text.strip() if soup.find("h1") else "Titre introuvable"

                # Date
                date_tag = soup.find("time")
                date_publication = date_tag.text.strip() if date_tag else "Date inconnue"

                # Texte intégral
                paragraphes = soup.find_all("p")
                texte = "\n".join([p.text.strip() for p in paragraphes if len(p.text.strip()) > 20])

                # Source
                source = "Cameroon Tribune"

                # Thème (facultatif, on peut extraire les mots-clés si dispo)
                meta_theme = soup.find("meta", {"name": "keywords"})
                theme = meta_theme["content"] if meta_theme else "Non défini"

                # Écriture dans le CSV
                writer.writerow([titre, texte[:5000], date_publication, source, url, theme])  # Texte limité à 1000 caractères

                print(f" Article extrait : {titre}")
            else:
                print(f" Erreur HTTP {response.status_code} pour {url}")
        except Exception as e:
            print(f" Erreur lors du traitement de {url} : {str(e)}")

        # Pause entre les requêtes
        time.sleep(random.uniform(1.5, 3.5))
