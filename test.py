import streamlit as st
# Titre de l'application
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
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
df4 = df4.drop(["Confidence in national government"], axis = 1)
df5 = df4.groupby(['Country name'])['Life Ladder'].mean().to_frame().reset_index()
df5 = df5.sort_values(by='Life Ladder', ascending = False)

# Modèles Ensemble : 
df2021 = df4[df4['year']==2021]
# Séparation de la variable cible des fonctionnalités
y_2021 = df2021['Life Ladder']
X_2021 = df2021.drop(columns=['Life Ladder'])
df4 = df4[df4['year']!=2021]
df_ensemble = df4.dropna(axis = 0, how = "any")
df_ensemble = df_ensemble.sort_values(by = "year", ascending = False)
df_ensemble = df_ensemble.drop("year", axis = 1)

def rmse_revu(y_true, y_pred, threshold = 0.3):
  errors = np.abs(y_true - y_pred)
  squared_errors = np.where(errors > threshold, errors**2 , 0)
  mean_squared_error = np.mean(squared_errors)
  final_score = np.sqrt(mean_squared_error)
  return final_score
    
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

# Modèles Lineaires : 
df_lineaire = df4.sort_values(by = "year", ascending = False)
df_lineaire = df_lineaire.drop("year", axis = 1)

# Division des données en ensembles d'entraînement et de test
yl = df_lineaire ['Life Ladder']
Xl = df_lineaire.drop(["Life Ladder"], axis = 1)
X_trainl, X_testl, y_trainl, y_testl = train_test_split(Xl, yl, test_size=0.3, random_state = 42)


# Prétraitement des colonnes numériques
numeric_transformerl = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Prétraitement des colonnes catégorielles
pays_transformerl = Pipeline(steps=[
    ('targetencoder', TargetEncoder())
])

# Prétraitement des colonnes catégorielles
region_transformerl = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Combiner les prétraitements numériques et catégoriels
preprocessorl = ColumnTransformer(
    transformers=[
        ('target', pays_transformerl, pays_cols),
        ('hotencoder', region_transformerl, region_cols),
        ('num', numeric_transformerl, numeric_cols)
    ])

#KNN
df_KNN = df4.sort_values(by = "year", ascending = False)
df_KNN = df_KNN.drop("year", axis = 1)

# Division des données en ensembles d'entraînement et de test
yk = df_KNN ['Life Ladder']
Xk = df_KNN.drop(["Life Ladder"], axis = 1)
X_traink, X_testk, y_traink, y_testk = train_test_split(Xk, yk, test_size=0.2, random_state = 42)
# Prétraitement des colonnes numériques
numeric_transformerk = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# Prétraitement des colonnes catégorielles
pays_transformerk = Pipeline(steps=[
    ('targetencoder', TargetEncoder())
])

# Prétraitement des colonnes catégorielles
region_transformerk = Pipeline(steps=[
    ('onehot', TargetEncoder())
])

# Combiner les prétraitements numériques et catégoriels
preprocessork = ColumnTransformer(
    transformers=[
        ('target', pays_transformerk, pays_cols),
        ('hotencoder', region_transformerk, region_cols),
        ('num', numeric_transformerk, numeric_cols)
    ])

#SVR
df_SVM = df4.sort_values(by = "year", ascending = False)
df_SVM = df_SVM.drop("year", axis = 1)

# Division des données en ensembles d'entraînement et de test
ys = df_SVM ['Life Ladder']
Xs = df_SVM.drop(["Life Ladder"], axis = 1)
X_trains, X_tests, y_trains, y_tests = train_test_split(Xs, ys, test_size=0.2, random_state = 42)

# Prétraitement des colonnes numériques
numeric_transformers = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Prétraitement des colonnes catégorielles
pays_transformers = Pipeline(steps=[
    ('targetencoder', TargetEncoder())
])
#OneHotEncoder
# Prétraitement des colonnes catégorielles
region_transformers = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Combiner les prétraitements numériques et catégoriels
preprocessors = ColumnTransformer(
    transformers=[
        ('target', pays_transformers, pays_cols),
        ('hotencoder', region_transformers, region_cols),
        ('num', numeric_transformers, numeric_cols)
    ])

# Création des pipelines pour les modèles
tree = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=42, max_depth=6))
random = make_pipeline(preprocessor, RandomForestRegressor(random_state=42, max_features=5))
adaboost = make_pipeline(preprocessor, AdaBoostRegressor(RandomForestRegressor(random_state=42, max_features=5, min_samples_split=2, n_estimators=300)))
linear = make_pipeline(preprocessorl, LinearRegression())
ridge = make_pipeline(preprocessorl, Ridge(alpha = 0.9 )) #corrélation entre les variables qui ne necessitent pas de revoir le poids des variables
lasso = make_pipeline(preprocessorl, Lasso(alpha = 0.1)) #avec un alpha a 0.1, le modèle ressemble à une régression linéaire standard ==> inutile car pas de grosses corélations entre les variables et pas de nécessité de mettre à 0 certaines variables
svr = make_pipeline(preprocessors, SVR(kernel = "rbf", C = 6))
boost = make_pipeline(preprocessors, AdaBoostRegressor(SVR(kernel = "rbf")))
knn = make_pipeline(preprocessork, KNeighborsRegressor(n_neighbors = 3, metric = "manhattan")) #optimisation du nombre de voisins

# Liste des modèles à entraîner
models = [
('Arbre de décision', tree),
('Random Forest', random),
('Linear Regression', linear),
('Ridge', ridge),
('Lasso', lasso),
('KNN', knn),
('SVR', svr)
]


st.title("Projet Bonheur")

# Barre latérale avec des options cliquables
option = st.sidebar.radio(
    'Menu',
    ('Présentation', 'Quelques visualisations', 'Pre-processing', 'Modélisation', "Modélisation nouvelles données")
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
    
    

    # Barre latérale pour choisir le modèle
    selected_model = st.sidebar.selectbox('Sélectionnez un modèle', [model_name for model_name, _ in models])
    st.header(selected_model)
    st.header("Modélisation jeu d'entraînement")
    # Entraînement du modèle sélectionné    
    for model_name, model in models:
        
        if model_name == selected_model:
            if model_name =='Arbre de décision' or model_name=='Random Forest':
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calcul des métriques
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.header("Prédictions 2021") 
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write(f"RMSE_REVU: {RMSE_REVU}")
                #Histogrammes des erreurs
                plt.figure(figsize = (8,4))
                err_hist = np.abs(y_2021 - y_pred_2021)
                err_hist_tol = []
                for erreur in err_hist:
                  if erreur >= 0.3:
                    err_hist_tol.append(erreur)
                plt.hist(err_hist_tol, label = model_name, bins = 5)
                plt.title("Histogramme des erreurs")
                plt.xticks([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4])
                plt.legend();
                st.pyplot(plt)
            elif model_name =='Linear Regression' or model_name == 'Ridge' or model_name == 'Lasso':
                model.fit(X_trainl, y_trainl)
                y_predl = model.predict(X_testl)

                # Calcul des métriques
                mae = mean_absolute_error(y_testl, y_predl)
                rmse = mean_squared_error(y_testl, y_predl, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.header("Prédictions 2021") 
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write(f"RMSE_REVU: {RMSE_REVU}")
                #Histogrammes des erreurs
                plt.figure(figsize = (8,4))
                err_hist = np.abs(y_2021 - y_pred_2021)
                err_hist_tol = []
                for erreur in err_hist:
                  if erreur >= 0.3:
                    err_hist_tol.append(erreur)
                plt.hist(err_hist_tol, label = model_name, bins = 5)
                plt.title("Histogramme des erreurs")

                plt.xticks([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4])
                plt.legend();
                st.pyplot(plt)
            elif model_name =='SVR' or model_name == 'BOOST':
                model.fit(X_trains, y_trains)
                y_preds = model.predict(X_tests)

                # Calcul des métriques
                mae = mean_absolute_error(y_tests, y_preds)
                rmse = mean_squared_error(y_tests, y_preds, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.header("Prédictions 2021") 
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write(f"RMSE_REVU: {RMSE_REVU}")
                #Histogrammes des erreurs
                plt.figure(figsize = (8,4))
                err_hist = np.abs(y_2021 - y_pred_2021)
                err_hist_tol = []
                for erreur in err_hist:
                  if erreur >= 0.3:
                    err_hist_tol.append(erreur)
                plt.hist(err_hist_tol, label = model_name, bins = 5)
                plt.title("Histogramme des erreurs")

                plt.xticks([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4])
                plt.legend();
                st.pyplot(plt)
            elif model_name =='KNN':
                model.fit(X_traink, y_traink)
                y_predk = model.predict(X_testk)

                # Calcul des métriques
                mae = mean_absolute_error(y_testk, y_predk)
                rmse = mean_squared_error(y_testk, y_predk, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.header("Prédictions 2021") 
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write(f"RMSE_REVU: {RMSE_REVU}")
                #Histogrammes des erreurs
                plt.figure(figsize = (8,4))
                err_hist = np.abs(y_2021 - y_pred_2021)
                err_hist_tol = []
                for erreur in err_hist:
                  if erreur >= 0.3:
                    err_hist_tol.append(erreur)
                plt.hist(err_hist_tol, label = model_name, bins = 5)
                plt.title("Histogramme des erreurs")

                plt.xticks([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4])
                plt.legend();
                st.pyplot(plt)
elif option == "Modélisation nouvelles données":
    # Création du formulaire
    with st.form('modélisation'):

        # Zone de liste avec une seule possibilité de sélection
        model_select = st.selectbox('Sélectionnez le modèle à utiliser', [model_name for model_name, _ in models])

        # Zone de liste avec une seule possibilité de sélection
        country_select = st.selectbox('Sélectionnez le pays', sorted(df_ensemble['Country name'].unique()))

        # Zones de texte pour les colonnes du DataFrame df_ensemble à partir de la deuxième colonne
        if "X_new" not in st.session_state:
            st.session_state.X_new = pd.dataframe()
        
        for column in list(df_ensemble.columns)[2:-1]:
            st.session_state.X_new[column] = st.text_input(column)

        # Bouton "Entraîner les modèles"
        submit_button = st.form_submit_button('Entraîner les modèles')
    
    if submit_button:

        # Création d'un nouveau jeu de données d'entraînement
        import pandas as pd
        X_train_new = pd.DataFrame(columns=list(df_ensemble.columns[:-1]))
        X_train_new['Country name'] = country_select
        #for column in list(df_ensemble.columns)[2:-1]:
        #    X_train_new[column] = st.session_state.X_new[column]
        
        #X_train_new = X_train_new.merge(df3, on='Country name')
        st.write(X_train_new)
        
# Pour exécuter l'application : streamlit run app.py
