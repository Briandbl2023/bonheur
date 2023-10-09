import streamlit as st

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

elif option == 'Quelques visualisations':
    st.header("Quelques visualisations du projet")
    st.write("Contenu de la présentation du projet.")

elif option == 'Pre-processing':
    st.header("Pre-Processing")
    st.write("Page pre-processing.")

elif option == 'Modélisation':
    st.header("Modélisation")
    st.write("Test de modélisation.")

# Pour exécuter l'application : streamlit run app.py
