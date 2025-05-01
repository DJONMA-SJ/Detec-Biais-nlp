import requests
import cloudscraper
from bs4 import BeautifulSoup
import time
import random
import csv

# Liste des 10 derniers liens d'articles politiques
urls = [
   "https://actucameroun.com/2025/04/10/cloture-de-la-premiere-session-ordinaire-de-lannee-legislative-discours-du-president-marcel-niat-njifendji/",
   "https://actucameroun.com/2025/04/10/mort-en-decembre-2024-les-obseques-du-chef-babone-auront-lieu-le-19-avril-2025/",
   "https://actucameroun.com/2025/04/10/affaire-diane-yangwo-shanda-tonme-exprime-sa-colere-au-juge-medou-dany-lor/",
   "https://actucameroun.com/2025/04/10/paul-biya-notre-republique-est-une-et-indivisible/",
   "https://actucameroun.com/2025/04/10/niger-franklin-nyamsi-recu-en-audience-par-le-general-tiani/",
   "https://actucameroun.com/2025/04/10/vincent-sosthene-fouda-revele-les-coulisses-dun-projet-politique-autour-de-marafa/",
   "https://actucameroun.com/2025/04/10/la-jeunesse-de-la-sanaga-maritime-jure-fidelite-a-jean-ernest-massena-ngalle-bibehe-et-son-candidat-paul-biya/",
   "https://actucameroun.com/2025/04/10/cavaye-yeguie-djibril-annonce-le-bapteme-du-palais-de-verre-paul-biya-pour-le-25-avril-prochain/",
   "https://actucameroun.com/2025/04/10/vincent-sosthene-fouda-explique-comment-le-mrc-a-ete-cree-pour-servir-de-porte-etendard-a-marafa/",
   "https://actucameroun.com/2025/04/09/ngaoundere-tolerance-zero-au-desordre-pour-la-fete-du-travail-2025/",
   "https://actucameroun.com/2025/04/09/nomination-de-monique-ouli-ndongo-au-conseil-constitutionnel-nkongho-felix-agbor-exprime-ses-inquietudes/",
]

# Simuler un navigateur pour éviter le blocage
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# Scraper qui gère Cloudflare
scraper = cloudscraper.create_scraper()

# Préparation du fichier CSV pour enregistrer les résultats
with open('articles_politique_actucameroun.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Titre", "Texte intégral", "Date", "Nom du média", "URL", "Thème"])

    # Traitement de chaque URL
    for url in urls:
        response = scraper.get(url, headers=headers)

        # Vérifier si la page est accessible
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")

            # Extraction du titre
            titre = soup.find("h1").text.strip() if soup.find("h1") else "Titre introuvable"

            # Extraction du texte intégral
            paragraphes = soup.find_all("p")
            texte_integral = "\n".join([p.text.strip() for p in paragraphes if len(p.text.strip()) > 20])

            # Extraction de la date de publication
            date_tag = soup.find("time")
            date_publication = date_tag.text.strip() if date_tag else "Date inconnue"

            # Extraction du nom du média (source)
            source_tag = soup.find("meta", {"name": "publisher"})  # Cherche la balise meta avec l'attribut 'publisher'
            source = source_tag["content"] if source_tag else "actu cameroun"

            # Extraction du thème (si disponible)
            theme_tag = soup.find("meta", {"name": "keywords"})  # Cherche la balise meta avec l'attribut 'keywords'
            theme = theme_tag["content"] if theme_tag else "Thème non disponible"

            # Écriture des résultats dans le fichier CSV
            writer.writerow([titre, texte_integral[:5000], date_publication, source, url, theme])  # Limite à 5000 caractères du texte intégral

            # Affichage des résultats pour suivi
            print(f"**Titre** : {titre}")
            print(f"**Date** : {date_publication}")
            print(f"**Nom du média** : {source}")
            print(f"**Texte intégral (extrait)** : {texte_integral[:5000]}...\n")
            print(f"**URL** : {url}\n")
        else:
            print(f"Erreur HTTP {response.status_code} pour l'URL : {url}")
        
        # Pause pour éviter le blocage par le site (ajouter un délai aléatoire entre les requêtes)
        time.sleep(random.uniform(1, 3))
