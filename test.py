import streamlit as st

# Titre de l'application
st.title("Application Streamlit avec Barre Latérale")

# Barre latérale avec 4 éléments
option = st.sidebar.selectbox(
    'Sélectionnez une option',
    ('Accueil', 'Page 1', 'Page 2', 'À propos de nous')
)

# Contenu en fonction de l'option sélectionnée
if option == 'Accueil':
    st.header("Bienvenue sur la page d'accueil")
    st.write("C'est la page principale de notre application.")

elif option == 'Page 1':
    st.header("Page 1")
    st.write("Contenu de la première page.")

elif option == 'Page 2':
    st.header("Page 2")
    st.write("Contenu de la deuxième page.")

elif option == 'À propos de nous':
    st.header("À propos de nous")
    st.write("Informations sur notre entreprise ou équipe.")

# Pour exécuter l'application : streamlit run app.py
