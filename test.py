import streamlit as st
!pip install xlrd
# Titre de l'application
st.title("Projet Bonheur")

# Barre latérale avec des options cliquables
option = st.sidebar.radio(
    'Sélectionnez une option',
    ('Présentation', 'Quelques visualisations', 'Pre-processing', 'Modélisation')
)

# Contenu en fonction de l'option sélectionnée
if option == 'Présentation':
    st.header("Présentation du projet")
    st.write("C'est la page de présentation du projet.")
    import pandas as pd

    # URL du fichier Excel sur GitHub
    github_url = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report-2022.xls'

    # Lire le fichier Excel dans un DataFrame
    df = pd.read_excel(github_url)

    # Maintenant, df contient les données du fichier Excel en tant que DataFrame
    print(df.head())  # Affiche les premières lignes du DataFrame


elif option == 'Quelques visualisations':
    st.header("Quelques visualisations du projet")
    st.write("Contenu de la présentation du projet.")
    import pandas as pd

    # URL du fichier Excel sur GitHub
    github_url = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report-2022.xls'

    # Lire le fichier Excel dans un DataFrame
    df = pd.read_excel(github_url)

    # Maintenant, df contient les données du fichier Excel en tant que DataFrame
    print(df.head())  # Affiche les premières lignes du DataFrame
    
elif option == 'Pre-processing':
    st.header("Pre-Processing")
    st.write("Page pre-processing.")

elif option == 'Modélisation':
    st.header("Modélisation")
    st.write("Test de modélisation.")

# Pour exécuter l'application : streamlit run app.py
