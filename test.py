import streamlit as st
# Titre de l'application
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikit-learn
from sklearn.pipeline import make_pipeline
from scikit-learn.ensemble import DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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
df5 = df4.groupby(['Country name'])['Life Ladder'].mean().to_frame().reset_index()
df5 = df5.sort_values(by='Life Ladder', ascending = False)
df_ensemble = df4.dropna(axis = 0, how = "any")
df_ensemble = df_ensemble.sort_values(by = "year", ascending = False)
df_ensemble = df_ensemble.drop("year", axis = 1)

# Division des données en ensembles d'entraînement et de test
y = df_ensemble ['Life Ladder']
X = df_ensemble.drop(["Life Ladder"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Colonnes catégorielles et numériques
pays_cols = ['Country name']
region_cols = ['Regional indicator']
numeric_cols = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
                'Positive affect', 'Negative affect', ]

# Prétraitement des colonnes numériques
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

# Prétraitement des colonnes catégorielles
pays_transformer = Pipeline(steps=[('targetencoder', TargetEncoder())])

# Prétraitement des colonnes catégorielles
region_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

# Combiner les prétraitements numériques et catégoriels
preprocessor = ColumnTransformer(transformers=[('target', pays_transformer, pays_cols),('hotencoder', region_transformer, region_cols),('num', numeric_transformer, numeric_cols)])

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

elif option == 'Quelques visualisations':
    st.header("Quelques visualisations du projet")

    # Créer le graphique avec Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    p = sns.barplot(y=df5['Country name'].head(10), x=df5['Life Ladder'].head(10))
    p.set_title("Top 10 des pays les plus heureux")
    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    plt.figure(figsize=(10, 6))
    p= sns.barplot(y=df5['Country name'].tail(10), x=df5['Life Ladder'].tail(10))
    p.set_title("Top 10 des pays les plus malheureux");
    st.pyplot(plt)
    
elif option == 'Pre-processing':
    st.header("Pre-Processing")
    st.write("Page pre-processing.")

elif option == 'Modélisation':
    st.header("Modélisation")
    st.write("Test de modélisation.")

    # Création des pipelines pour les modèles
    tree = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=42, max_depth=6))
    random = make_pipeline(preprocessor, RandomForestRegressor(random_state=42, max_features=5))
    adaboost = make_pipeline(preprocessor, AdaBoostRegressor(RandomForestRegressor(random_state=42, max_features=5, min_samples_split=2, n_estimators=300)))

    # Liste des modèles à entraîner
    models = [
    ('Arbre de décision', tree),
    ('Random Forest', random),
    ('Adaboost', adaboost)
    ]

    # Barre latérale pour choisir le modèle
    selected_model = st.sidebar.selectbox('Sélectionnez un modèle', [model_name for model_name, _ in models])

    # Entraînement du modèle sélectionné    
    for model_name, model in models:
        if model_name == selected_model:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calcul des métriques
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            # Affichage des résultats
            st.write(f"Modèle: {model_name}")
            st.write(f"MAE: {mae}")
            st.write(f"RMSE: {rmse}")



# Pour exécuter l'application : streamlit run app.py
