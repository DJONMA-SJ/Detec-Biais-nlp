import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Sidebar pour navigation
st.sidebar.title("Détection de Biais dans des corpus médiatiques")
page = st.sidebar.selectbox("Choisissez une section", [
    "Accueil",
  "Analyse émotionnelle",
    "Biais de source",
    "Biais de cadrage",
    "À propos"
])

# Charger les données une seule fois
@st.cache_data
def load_data():
    df = pd.read_json("data/Processed/polarite_cameroun_rss.json")
    return df

articles = load_data()

# Page d'accueil
if page == "Accueil":
    st.title("Application de Détection de Biais dans les Médias Camerounais")
    st.markdown("""
    Cette application analyse automatiquement des articles de presse pour identifier :
    - les **émotions** exprimées,
    - la **polarité** du discours,
    - les biais de **source**, **cadrage** et **omission**.

    Données : Corpus politique camerounais
    """)

# Page émotion
elif page == "Analyse émotionnelle":
    st.title("Analyse Émotionnelle")
    st.markdown("Visualisation des émotions détectées par média et dans le temps.")

    # Exemple d'affichage : intensité émotionnelle par média
    st.subheader("Intensité émotionnelle par média")
    fig, ax = plt.subplots(figsize=(10, 5))
    articles.boxplot(column="proportion_émotion", by="Nom du média", ax=ax, rot=45)
    st.pyplot(fig)

# Page biais de source
elif page == "Biais de source":
    st.title("Analyse du Biais de Source")
    st.markdown("Polarité moyenne et distribution des sentiments par média.")
    # (Insérer ici tes visualisations de biais de source)

# Page biais de cadrage
elif page == "Biais de cadrage":
    st.title("Analyse du Biais de Cadrage")
    st.markdown("Analyse des mots à polarité autour des entités nommées.")
    # (À intégrer plus tard)

# Page à propos
elif page == "À propos":
    st.title("À propos")
    st.markdown("""
    **Projet universitaire / de recherche**

    Réalisé par : [Josephine ]  
    Encadré par :Mr Samdalle Amaria
    Sources : ...
    """)
