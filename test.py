import streamlit as st
# Titre de l'application
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# URL du fichier Excel sur GitHub
github_url = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report-2022.xls'
github_url2 = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report-2021.csv'
    
# Lire le fichier Excel dans un DataFrame
df = pd.read_excel(github_url)
df = df[df['year']!=2005]
df1 = pd.read_csv(github_url2)
df3=pd.DataFrame()
df3['Country name']= df1['Country name']
df3['Regional indicator'] = df1['Regional indicator']
df4 = df.merge(df3, on='Country name')
# Maintenant, df contient les données du fichier Excel en tant que DataFrame

st.title("Projet Bonheur")

# Barre latérale avec des options cliquables
option = st.sidebar.radio(
    'Menu',
    ('Présentation', 'Quelques visualisations', 'Pre-processing', 'Modélisation')
)

# Contenu en fonction de l'option sélectionnée
if option == 'Présentation':
    st.header("Présentation du projet")
    st.write("C'est la page de présentation du projet.")
        
    st.write(df4.head())  # Affiche les premières lignes du DataFrame
    st.write(df4.isna().sum())

elif option == 'Quelques visualisations':
    st.header("Quelques visualisations du projet")

    # Créer le graphique avec Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    p = sns.barplot(y=df5['Country name'].head(10), x=df5['Life Ladder'].head(10))
    p.set_title("Top 10 des pays les plus heureux")

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

elif option == 'Pre-processing':
    st.header("Pre-Processing")
    st.write("Page pre-processing.")

elif option == 'Modélisation':
    st.header("Modélisation")
    st.write("Test de modélisation.")

# Pour exécuter l'application : streamlit run app.py
