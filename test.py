import streamlit as st

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
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder
# URL du fichier Excel sur GitHub
github_url = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report-2022.xls'
github_url2 = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report-2021.csv'
github_url3 = 'https://github.com/Briandbl2023/bonheur/raw/main/world-happiness-report_1.csv'
logods ='https://github.com/Briandbl2023/bonheur/raw/main/logo-2021.png'
pknn = 'https://github.com/Briandbl2023/bonheur/raw/main/knn_2.jpg'
psvr = 'https://github.com/Briandbl2023/bonheur/raw/main/svr_2.jpg'
parbre = 'https://github.com/Briandbl2023/bonheur/raw/main/arbre_preprocessing.jpg'
plineaire = 'https://github.com/Briandbl2023/bonheur/raw/main/lineaire_preprocessing.jpg'

oknn = 'https://github.com/Briandbl2023/bonheur/raw/main/knn_optimisation.jpg'
osvr = 'https://github.com/Briandbl2023/bonheur/raw/main/svr_optimisation.jpg'
oarbre = 'https://github.com/Briandbl2023/bonheur/raw/main/arbre_optimisation.jpg'
orandom = 'https://github.com/Briandbl2023/bonheur/raw/main/random_optimisation.jpg'
olineaire = 'https://github.com/Briandbl2023/bonheur/raw/main/lineaire_optimisation.jpg'
oridge = 'https://github.com/Briandbl2023/bonheur/raw/main/ridge_optimisation.jpg'
olasso = 'https://github.com/Briandbl2023/bonheur/raw/main/lasso_opitmisation.jpg'

# Lire le fichier Excel dans un DataFrame
df = pd.read_csv(github_url3)
df = df[df['year']!=2005]
df1 = pd.read_csv(github_url2)
df2 = pd.read_excel(github_url)
df3=pd.DataFrame()
df3['Country name']= df1['Country name']
df3['Regional indicator'] = df1['Regional indicator']
df4 = df.merge(df3, on='Country name')
df2 = df2.drop(["Confidence in national government"], axis = 1)
df5 = df4.groupby(['Country name'])['Life Ladder'].mean().to_frame().reset_index()
df5 = df5.sort_values(by='Life Ladder', ascending = False)
df2021 = df2.merge(df3, on='Country name')
# Modèles Ensemble : 
df2021 = df2021[df2021['year']==2021]
nb_pays_region = df2021["Regional indicator"].value_counts()
# Séparation de la variable cible des fonctionnalités
y_2021 = df2021['Life Ladder']
X_2021 = df2021.drop(columns=['Life Ladder'])
df_ensemble = df4.sort_values(by = "year", ascending = False)
df_ensemble = df_ensemble.drop("year", axis = 1)

#def rmse_revu(y_true, y_pred, threshold = 0.3):
#  errors = np.abs(y_true - y_pred)
#  squared_errors = np.where(errors > threshold, errors**2 , 0)
#  mean_squared_error = np.mean(squared_errors)
#  final_score = np.sqrt(mean_squared_error)
#  return final_score

#Modèles ensembles
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
K_MEANS = df
K_MEANS = K_MEANS.sort_values(by = "year", ascending = False)
K_MEANS = K_MEANS.drop("year", axis=1)
#Récupération des clusters via un K_means
K_MEANS = K_MEANS.groupby("Country name").mean()

# Récupération des features
features = K_MEANS.drop(columns = 'Life Ladder')


#Variables à standardiser
standard_col = ['Log GDP per capita', 'Social support', 'Freedom to make life choices', "Healthy life expectancy at birth",
                'Perceptions of corruption', 'Positive affect','Negative affect']

# Prétraitement des colonnes numériques
numeric_transformerkm = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# Combiner les prétraitements pour toutes les colonnes
preprocessorkm = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, standard_col)
    ])

#application
from sklearn.pipeline import make_pipeline
preprocessing = make_pipeline(preprocessorkm)
features_normalises = preprocessing.fit_transform(features)
kmeans = KMeans(n_clusters = 4)
kmeans.fit(features_normalises)
features_predict = kmeans.predict(features_normalises)

#récupérer les coordonnées de tous les points par k-means
groupe_1_index = features[features_predict == 0].index
groupe_2_index = features[features_predict == 1].index
groupe_3_index = features[features_predict == 2].index
groupe_4_index = features[features_predict == 3].index
#groupe_5_index = features[features_predict == 4].index

#création d'un dictionnaire
groupe_1 = [1]* groupe_1_index.shape[0]
groupe_1_dico = dict(zip(groupe_1_index, groupe_1))

groupe_2 = [2]* groupe_2_index.shape[0]
groupe_2_dico = dict(zip(groupe_2_index, groupe_2))

groupe_3 = [3]* groupe_3_index.shape[0]
groupe_3_dico = dict(zip(groupe_3_index, groupe_3))

groupe_4 = [4]* groupe_4_index.shape[0]
groupe_4_dico = dict(zip(groupe_4_index, groupe_4))

#groupe_5 = [5]* groupe_5_index.shape[0]
#groupe_5_dico = dict(zip(groupe_5_index, groupe_4))

dico_groupe = (groupe_1_dico | groupe_2_dico | groupe_3_dico | groupe_4_dico)


df_lineaire = df4.sort_values(by = "year", ascending = False)
df_lineaire = df_lineaire.drop("year", axis = 1)
df_lineaire["k_means"] = df_lineaire["Country name"]
df_lineaire["k_means"] = df_lineaire["k_means"].replace(dico_groupe)
df_lineaire["k_means"] = df_lineaire["k_means"].apply(lambda x : str(x))
# Division des données en ensembles d'entraînement et de test
yl = df_lineaire ['Life Ladder']
Xl = df_lineaire.drop(["Life Ladder"], axis = 1)
X_trainl, X_testl, y_trainl, y_testl = train_test_split(Xl, yl, test_size=0.3, random_state = 42)
def gestion_nan1(X):
  for colonne in numeric_cols:
    if 'Regional indicator' in X.columns:
      X[colonne] = X[colonne].fillna(X.groupby("Regional indicator")[colonne].transform('median'))

  X_new = X #.drop("Regional indicator", axis = 1)

  return X_new

X_trainl = gestion_nan1(X_trainl)
X_testl = gestion_nan1(X_testl)

k_means_cols = ['k_means']

# Prétraitement des colonnes numériques
numeric_transformerl = Pipeline(steps=[
    ('scaler', StandardScaler())
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
        ('hotencoder', region_transformerl, k_means_cols),
        ('num', numeric_transformerl, numeric_cols)
    ])

#KNN
df_KNN = df4.sort_values(by = "year", ascending = False)
df_KNN = df_KNN.drop("year", axis = 1)

# Division des données en ensembles d'entraînement et de test
yk = df_KNN ['Life Ladder']
Xk = df_KNN.drop(["Life Ladder"], axis = 1)
X_traink, X_testk, y_traink, y_testk = train_test_split(Xk, yk, test_size=0.1, random_state = 42)
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
X_trains = gestion_nan1(X_trains)
X_tests = gestion_nan1(X_tests)
# Prétraitement des colonnes numériques
numeric_transformers = Pipeline(steps=[
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
tree = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=42, max_depth=6, min_samples_split = 3))
random = make_pipeline(preprocessor, RandomForestRegressor(random_state=42, max_depth=16, min_samples_split = 7))
linear = make_pipeline(preprocessorl, LinearRegression())
ridge = make_pipeline(preprocessorl, Ridge(alpha = 1, solver = "sag")) #corrélation entre les variables qui ne necessitent pas de revoir le poids des variables
lasso = make_pipeline(preprocessorl, Lasso(alpha = 0.1)) #avec un alpha a 0.1, le modèle ressemble à une régression linéaire standard ==> inutile car pas de grosses corélations entre les variables et pas de nécessité de mettre à 0 certaines variables
svr = make_pipeline(preprocessors, SVR(C = 15))
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

# Liste des modèles à entraîner
modelsp = [
('KNN', knn),
('SVR', svr)
]
# Barre latérale avec des options cliquables
# Titre en gras dans la barre latérale
#st.sidebar.markdown("<b>Sommaire</b>", unsafe_allow_html=True)

# Utilisation de la chaîne de caractères comme label
option = st.sidebar.radio(
    '',  # Utilisation d'un espace insécable comme label
    ('Contexte', 'Exploration', 'Modélisation', "Prédictions", "Conclusion"),
  #label_position="hidden"
)


#st.sidebar.write("<b>Sommaire</b>",unsafe_allow_html=True)
#option = st.sidebar.radio(
#    '\u200B',('Contexte', 'Exploration', 'Modélisation', "Prédictions", "Conclusion")
#)
about = "<b>About</b><br>Projet fil rouge dans le cadre de la formation Data Analyst sur le rapport mondial du bien être, publié tous les ans par l'ONU"
st.sidebar.markdown(about, unsafe_allow_html=True)
auteurs = "<b>Auteurs</b><br>Gaëlle Ekindi<br>Jihade El Ouardi<br>Patricia Verissimo<br>Stéphane Burel<br><br>"
st.sidebar.markdown(auteurs, unsafe_allow_html=True)
st.sidebar.image(logods)

# Contenu en fonction de l'option sélectionnée
if option == 'Contexte':
    st.header("Analyse du bonheur")
    st.header("Présentation du projet")
    st.write("C'est la page de présentation du projet.")
        #Happiness rankings of nations in 2017

    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, iplot, plot
    import pandas as pd
    df_2017 = df[df['year']!=2017]
    df_2017['RANK'] = df_2017['Life Ladder'].rank()
    #init_notebook_mode(connected=True)

    data = dict(
        type='choropleth',
        locations=df_2017['Country name'],
        locationmode='country names',
        z=df_2017['RANK'],
        text=df_2017['Country name'],
        colorbar={'title': 'Happiness'}
    )

    layout = dict(
        title='Global Happiness 2017',
        geo=dict(showframe=False)
    )

    choromap3 = go.Figure(data=[data], layout=layout)

    st.plotly_chart(choromap3)

    st.write(df4.head())  

elif option == 'Exploration':
    st.header("Quelques visualisations du projet")

    # Créer le graphique avec Seaborn
    sns.set(style="whitegrid")
    #graphique de densité kernel (KDE)

    plt.figure(figsize=(15,8))
    sns.kdeplot(x=df4["Life Ladder"], hue=df4["Regional indicator"],fill=True ,linewidth=2)
    plt.axvline(df4["Life Ladder"].mean(),c="black")
    plt.title("Graphique de densité Kernel (KDE)")
    st.pyplot(plt)

    # Sélectionner uniquement les colonnes numériques du DataFrame
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64'])

    # Calculer la matrice de corrélation pour les colonnes numériques
    cor = round(colonnes_numeriques.corr(), 2)
    # Créer un heatmap à l'aide de seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
    plt.title("Correlations avec Life Ladder")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(plt)
  
    #plt.figure(figsize=(10, 6))
    #p = sns.barplot(y=df5['Country name'].head(10), x=df5['Life Ladder'].head(10))
    #p.set_title("Top 10 des pays les plus heureux")
    # Afficher le graphique dans Streamlit
    #st.pyplot(plt)
    #plt.figure(figsize=(10, 6))
    #p= sns.barplot(y=df5['Country name'].tail(10), x=df5['Life Ladder'].tail(10))
    #p.set_title("Top 10 des pays les plus malheureux");
    #st.pyplot(plt)
    
elif option == 'Modélisation':
    
    st.header("Modélisation")
    textem = "<b><u>Les enjeux de la modélisation : </u></b><ul><li>Cerner les meilleurs paramétrages de pré-processing</li><li>Vérifier la robustesse de nos algorithmes</li><li>Trouver l’algorithme qui s’adaptera au mieux aux relations complexes de notre jeu de données</li></ul>"
    st.markdown(textem, unsafe_allow_html=True)

    # Barre latérale pour choisir le modèle
    selected_model = st.selectbox('Sélectionnez un modèle', [model_name for model_name, _ in models])
    #st.header(selected_model)
    st.write("<b><u>Pré-processing</b></u>",unsafe_allow_html=True)
    # Entraînement du modèle sélectionné    
    for model_name, model in models:
        
        if model_name == selected_model:
            if model_name =='Arbre de décision' or model_name=='Random Forest':
                st.image(parbre)
                st.write("<b><u>Optimisation</b></u>",unsafe_allow_html=True)
                if model_name=='Arbre de décision':
                  st.image(oarbre)
                elif model_name=='Random Forest':
                  st.image(orandom)
              
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calcul des métriques
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                #st.write(f"RMSE_REVU: {RMSE_REVU}")
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
                
                #residus = y_2021 - y_pred_2021
                #plt.figure(figsize=(8, 6))
                #plt.hist(residus, bins=30, color='blue', alpha=0.7)
                #plt.xlabel('Résidus')
                #plt.ylabel('Fréquence')
                #plt.title('Histogramme des Résidus')
                #st.pyplot(plt)
                ecart_optim = abs(y_pred_2021 - y_2021)
                ecart_optim_df = ecart_optim.to_frame(name='Ecarts')
                ecart_optim_df["Regional indicator"] = X_2021["Regional indicator"]
                bonnes_reponses = ecart_optim_df[ecart_optim_df["Ecarts"] < 0.3]
                bonnes_reponses = bonnes_reponses["Regional indicator"].value_counts()
                pourcentage_bonnes_reponses = ((bonnes_reponses/nb_pays_region)*100).sort_values(ascending = False)
                st.write("<b><u>Taux de réussite par région (exprimé en pourcentage)</u></b>",unsafe_allow_html=True)
                st.write(pourcentage_bonnes_reponses)
            elif model_name =='Linear Regression' or model_name == 'Ridge' or model_name == 'Lasso':
                st.image(plineaire)
                st.write("<b><u>Optimisation</b></u>",unsafe_allow_html=True)
                if model_name=='Linear Regression':
                  st.image(olineaire)
                elif model_name=='Ridge':
                  st.image(oridge)
                elif model_name=='Lasso':
                  st.image(olasso)
                model.fit(X_trainl, y_trainl)
                y_predl = model.predict(X_testl)
                X_2021l = gestion_nan1(X_2021)
                X_2021l["k_means"] = X_2021l["Country name"]
                X_2021l["k_means"] = X_2021l["k_means"].replace(dico_groupe)
                X_2021l["k_means"] = X_2021l["k_means"].apply(lambda x : str(x))
                # Calcul des métriques
                mae = mean_absolute_error(y_testl, y_predl)
                rmse = mean_squared_error(y_testl, y_predl, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")                
                st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021l)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                #st.write(f"RMSE_REVU: {RMSE_REVU}")
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
                
                #residus = y_2021 - y_pred_2021
                #plt.figure(figsize=(8, 6))
                #plt.hist(residus, bins=30, color='blue', alpha=0.7)
                #plt.xlabel('Résidus')
                #plt.ylabel('Fréquence')
                #plt.title('Histogramme des Résidus')
                #st.pyplot(plt)
                ecart_optim = abs(y_pred_2021 - y_2021)
                ecart_optim_df = ecart_optim.to_frame(name='Ecarts')
                ecart_optim_df["Regional indicator"] = X_2021["Regional indicator"]
                bonnes_reponses = ecart_optim_df[ecart_optim_df["Ecarts"] < 0.3]
                bonnes_reponses = bonnes_reponses["Regional indicator"].value_counts()
                pourcentage_bonnes_reponses = ((bonnes_reponses/nb_pays_region)*100).sort_values(ascending = False)
                st.write("<b><u>Taux de réussite par région (exprimé en pourcentage)</u></b>",unsafe_allow_html=True)
                st.write(pourcentage_bonnes_reponses)
            elif model_name =='SVR' or model_name == 'BOOST':
                st.image(psvr)
                st.write("<b><u>Optimisation</b></u>",unsafe_allow_html=True)
                st.image(osvr)
                model.fit(X_trains, y_trains)
                y_preds = model.predict(X_tests)
                X_2021s = gestion_nan1(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_tests, y_preds)
                rmse = mean_squared_error(y_tests, y_preds, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021s)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                #st.write(f"RMSE_REVU: {RMSE_REVU}")
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
                
                #residus = y_2021 - y_pred_2021
                #plt.figure(figsize=(8, 6))
                #plt.hist(residus, bins=30, color='blue', alpha=0.7)
                #plt.xlabel('Résidus')
                #plt.ylabel('Fréquence')
                #plt.title('Histogramme des Résidus')
                #st.pyplot(plt)
                ecart_optim = abs(y_pred_2021 - y_2021)
                ecart_optim_df = ecart_optim.to_frame(name='Ecarts')
                ecart_optim_df["Regional indicator"] = X_2021["Regional indicator"]
                bonnes_reponses = ecart_optim_df[ecart_optim_df["Ecarts"] < 0.3]
                bonnes_reponses = bonnes_reponses["Regional indicator"].value_counts()
                pourcentage_bonnes_reponses = ((bonnes_reponses/nb_pays_region)*100).sort_values(ascending = False)
                st.write("<b><u>Taux de réussite par région (exprimé en pourcentage)</u></b>",unsafe_allow_html=True)
                st.write(pourcentage_bonnes_reponses)
            elif model_name =='KNN':
                st.image(pknn)
                st.write("<b><u>Optimisation</b></u>",unsafe_allow_html=True)
                st.image(oknn)
                model.fit(X_traink, y_traink)
                y_predk = model.predict(X_testk)

                # Calcul des métriques
                mae = mean_absolute_error(y_testk, y_predk)
                rmse = mean_squared_error(y_testk, y_predk, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_pred_2021, y_2021)
                rmse = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")
                #st.write(f"RMSE_REVU: {RMSE_REVU}")
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
                
                #residus = y_2021 - y_pred_2021
                #plt.figure(figsize=(8, 6))
                #plt.hist(residus, bins=30, color='blue', alpha=0.7)
                #plt.xlabel('Résidus')
                #plt.ylabel('Fréquence')
                #plt.title('Histogramme des Résidus')
                #st.pyplot(plt)
                ecart_optim = abs(y_pred_2021 - y_2021)
                ecart_optim_df = ecart_optim.to_frame(name='Ecarts')
                ecart_optim_df["Regional indicator"] = X_2021["Regional indicator"]
                bonnes_reponses = ecart_optim_df[ecart_optim_df["Ecarts"] < 0.3]
                bonnes_reponses = bonnes_reponses["Regional indicator"].value_counts()
                pourcentage_bonnes_reponses = ((bonnes_reponses/nb_pays_region)*100).sort_values(ascending = False)
                st.write("<b><u>Taux de réussite par région (exprimé en pourcentage)</u></b>",unsafe_allow_html=True)
                st.write(pourcentage_bonnes_reponses)

elif option == "Prédictions":
    st.header("Prédictions")
    st.write("Le k-NN et le SVR sont les deux algorithmes qui captent le mieux les relations complexes de notre jeu de données : l’un fait moins de petites erreurs et l’autre fait moins de grosses erreurs.")
    st.write("Nous vous proposons, à partir de données totalement inconnues, de réaliser quelques prédictions sur ces deux modèles préalablement entraînés sur le jeu de données 2006-2020.") 
    st.write("Pour le bon fonctionnement de ces derniers, vous devrez : ")
    st.write("<ul><li>sélectionner un des pays dans la liste des 166 pays analysés</li><li>saisir un nombre entre 0 et 11 pour le champ 'Log GDP per capita'</li><li>saisir un âge entre 0 et 100 ans pour le champ 'Healthy life expectancy at birth</li><li>saisir un chiffre décimal compris entre -1 et 1 pour le champ 'Generosity'</li><li>saisir un chiffre décimal entre 0 et 1 pour les autres variables</li></ul>", unsafe_allow_html=True)     
    #st.write("Certaines variables numériques pourront être omises. Elles seront remplacées par la moyenne de la région (SVR) ou un imputer (k-NN).")
    #st.write("L’ajout de la région, le remplacement des valeurs manquantes ou la mise à l’échelle sera réalisé en automatique à l’aide du code et d’une pipeline.")
  # Création du formulaire
    with st.form('modélisation'):
        # Zone de liste avec une seule possibilité de sélection
        if "model_select" not in st.session_state :
            st.session_state.model_select=""
        st.session_state.model_select = st.selectbox('Sélectionnez le modèle à utiliser', [model_name for model_name, _ in modelsp])
                   
        #  Zone de liste avec une seule possibilité de sélection
        if "country_select" not in st.session_state :
            st.session_state.country_select=""
        st.session_state.country_select = st.selectbox('Sélectionnez le pays', sorted(df_ensemble['Country name'].unique()))

        # Dictionnaire pour stocker les valeurs saisies par l'utilisateur
        user_inputs = {}

        # Zones de texte pour les colonnes du DataFrame df_ensemble à partir de la deuxième colonne
        for column in list(df_ensemble.columns)[2:-1]:
            user_inputs[column] = st.text_input(column)

        # Bouton "Entraîner les modèles"
        submit_button = st.form_submit_button('Prédiction Life Ladder')

    # Création du DataFrame à partir des valeurs saisies par l'utilisateur
    if submit_button:
        selected_model = st.session_state.model_select
        # Filtrer les valeurs non vides
        user_inputs = {key: value for key, value in user_inputs.items() if value != ''}
        user_inputs = {clef: float(valeur.replace(',', '.')) for clef, valeur in user_inputs.items()}

        # Créer un DataFrame à partir du dictionnaire
        X_new = pd.DataFrame([user_inputs])
       
        # Création d'un nouveau jeu de données d'entraînement
        X_train_new = pd.DataFrame(columns=list(df_ensemble.columns[:-1]))
        X_train_new = X_train_new.drop(columns=['Life Ladder'])
        
        #st.session_state.country_select
        for column in X_train_new:
            if column != 'Country name':
                X_train_new[column] = X_new[column]
        
        X_train_new.loc[0, 'Country name'] = st.session_state.country_select
        
        X_train_new = X_train_new.merge(df3, on='Country name')
        
        for model_name, model in modelsp:
        # Création des pipelines pour les modèles

            if model_name == selected_model:
                
                if model_name =='SVR' or model_name == 'BOOST':
                    model.fit(X_trains, y_trains)
                    y_preds = model.predict(X_tests)
                    y_pred_saisie = model.predict(X_train_new)
                    st.write(y_pred_saisie)
                    st.header("Prédiction : " + str(y_pred_saisie[0]))
            
                elif model_name =='KNN':
                    model.fit(X_traink, y_traink)
                    y_predk = model.predict(X_testk)
                    y_pred_saisie = model.predict(X_train_new)
                    st.header("Prédiction : " + str(y_pred_saisie[0]))
        st.write("Dataframe :")
        st.write(X_train_new)
elif option == "Conclusion":
    st.write("<u>Les résultats</u><ul><li>Les résultats sont impressionnants par rapport aux doutes que nous avions sur la précision des indicateurs subjectifs.</li><li>Avec le SVR ou le k-NN, nous avons un taux de réussite d’environ 70% qui pourrait peut-être être amélioré avec l’introduction de nouvelles variables.</li></ul>",unsafe_allow_html=True)
    st.write("<u>Les limites</u><ul><li>L’analyse prédictive ne nous permet pas d’identifier les liens de causalité entre les différentes variables.</li><li>Il est difficile de pouvoir déterminer des leviers pour améliorer le bonheur dans chaque pays.</li></ul>", unsafe_allow_html=True)

# Pour exécuter l'application : streamlit run app.py
