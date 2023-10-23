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
dfj = pd.read_csv(github_url3)
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

def gestion_nan2(X, Y):
  X.columns = X.columns.str.strip()  # Supprimer les espaces en début et fin de nom de colonne
  Y.columns = Y.columns.str.strip()

  # Fusionner les DataFrames sur la colonne 'Regional indicator' en respectant la condition
  merged_df = pd.merge(X, Y, on='Regional indicator', how='left')
  merged_df = merged_df[merged_df['Country name'] == merged_df['Country name_y']]  # Appliquer la condition

  # Réinitialiser l'index et supprimer l'ancien index
  merged_df.reset_index(drop=True, inplace=True)
  st.write(merged_df)
  for colonne in numeric_cols:
    X[colonne] = X[colonne].fillna(merged_df[colonne + '_y'].median())

  # Supprimer les colonnes ajoutées lors de la fusion si nécessaire
  X.drop(columns=[colonne + '_y' for colonne in numeric_cols], inplace=True)


  return X

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
# Prétraitement des colonnes catégorielles
pays_transformerk2 = Pipeline(steps=[
    ('targetencoder', OneHotEncoder())
])

# Prétraitement des colonnes catégorielles
region_transformerk2 = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Combiner les prétraitements numériques et catégoriels
preprocessork2 = ColumnTransformer(
    transformers=[
        ('target', pays_transformerk2, pays_cols),
        ('hotencoder', region_transformerk2, region_cols),
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

# Prétraitement des colonnes catégorielles
pays_transformers2 = Pipeline(steps=[
    ('targetencoder', OneHotEncoder())
])

# Combiner les prétraitements numériques et catégoriels
preprocessors2 = ColumnTransformer(
    transformers=[
        ('target', pays_transformers2, pays_cols),
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
svr2 = make_pipeline(preprocessors2, SVR(C = 15))
knn2 = make_pipeline(preprocessork2, KNeighborsRegressor(n_neighbors = 3, metric = "manhattan")) #optimisation du nombre de voisins

# Liste des modèles à entraîner
models = [
('Arbre de décision', tree),
('Random Forest', random),
('Linear Regression', linear),
#('Ridge', ridge),
#('Lasso', lasso),
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
    st.header("Contexte")
    st.write("À quel point les gens sont-ils heureux aujourd'hui ?")
    st.write("Dans quelle mesure les gens sont-ils satisfaits de leur vie dans différentes sociétés ?") 
    st.write("Et comment nos conditions de vie affectent-elles tout cela ?")
    st.write("<b><u>Objectifs : </u></b><ul><li>Etudier les principaux déterminants du bonheur</li><li>Identifier des stratégies d'intervention pour améliorer la qualité de vie des individus</li><li>Prédire avec précision notre variable cible pour l'année 2021 en utilisant les données de 2005 à 2020 et en appliquant le modèle le plus approprié</li><li>Evaluer la pertinence des résultats obtenus à partir du World Happiness Report</li><li>Créer et présenter des visualisations intéractives</li></ul>", unsafe_allow_html=True)     

    #Happiness rankings of nations in 2017

    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, iplot, plot
    import pandas as pd
    selected_columns = ["year", "Life Ladder", "Country name"]
    df_map = dfj[selected_columns]
    data = []

    for year in range(2005, 2021):
        year_data = df_map[df_map['year'] == year]

        data.append(
            go.Choropleth(
                locations=year_data['Country name'],
                locationmode='country names',
                z=year_data['Life Ladder'],
                text=year_data['Country name'],
                colorbar={'title': 'Happiness'},
                showscale= True
            )
        )

    sliders = [{
        'active': 10,  # L'année active au début
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Year:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'steps': []
    }]

    for i, year in enumerate(range(2005, 2021)):
        step = {
            'args': [
                ['visible'],
                [False] * len(data)
            ],
            'label': str(year),
            'method': 'animate'
        }
        step['args'][1][i] = True
        sliders[0]['steps'].append(step)

    layout = dict(
        title='Global Happiness by Year (2005-2020)',
        geo=dict(showframe=False),
        sliders=sliders
    )
    # Créez la figure animée
    import plotly.express as px

    # Triez le DataFrame par la colonne "year"
    df_map = df_map.sort_values(by='year')

    fig = px.choropleth(
        df_map,
        locations='Country name',
        locationmode='country names',
        color='Life Ladder',
        animation_frame='year',  # Utilisez "year" en minuscules
        color_continuous_scale='RdYlGn',
        range_color=[df_map['Life Ladder'].min(), df_map['Life Ladder'].max()],
        title='Global Happiness by Year (2005-2020)'
    )

    fig.update_geos(showcoastlines=False, projection_type="natural earth")
    st.plotly_chart(fig)

    st.write("<b><u>Evolution au fil des années</u></b><br>",unsafe_allow_html=True)
    plt.figure(figsize = (15, 6))
    plt.title("Evolution du bien-être moyen par région du monde \n")
    plt.xlim(2005, 2020)
    evolution_life_ladder = df4.groupby(["Regional indicator", "year"], as_index = False).agg({"Life Ladder" : "median"})
    evolution_life_ladder["year"] = evolution_life_ladder["year"].astype("float")
    sns.lineplot(x = "year", y = "Life Ladder", data = evolution_life_ladder, hue = "Regional indicator", errorbar = None)
    plt.legend(bbox_to_anchor = (1,1))
    st.pyplot(plt)
  
    #st.write(df4.head())  

elif option == 'Exploration':
    st.header("Exploration de données et Data visualisation")
    st.write("<b><u>Le jeu de données : </u></b><ul><li>Le premier jeu de données contient les analyses de 166 pays pour la période de 2005 à 2020, totalisant ainsi 1949 lignes d'enregistrement", unsafe_allow_html=True)
    st.write(dfj.head(10))
    st.write("</li></ul><ul><li>Le second jeu de donnes se concentre exclusivement sur l'année 2021 et concerne 149 pays, correspondant à 149 lignes d'enregistrement<br>", unsafe_allow_html=True)
    st.write(df1.head(10))
    st.write("</li></ul>", unsafe_allow_html=True)     

    st.write("<b><u>Description des variables</u></b>",unsafe_allow_html=True)
    # Obtenez la liste des noms des colonnes
    
    colonnes = df.columns[1:].tolist()
    
    # Affichez la liste déroulante dans Streamlit
    colonne_selectionnee = st.selectbox("Sélectionnez une variable", colonnes)

    # Créez votre graphique en utilisant la colonne sélectionnée
    plt.figure(figsize=(8, 6))
    sns.histplot(dfj[colonne_selectionnee], bins=20, kde=True)
    plt.title(f'Distribution de {colonne_selectionnee}')
    plt.xlabel(colonne_selectionnee)
    plt.ylabel('Fréquence')
    plt.grid(True)
    st.pyplot(plt)  # Affichez le graphique dans Streamlit
    
    # Affichez les statistiques descriptives de la colonne sélectionnée
    #st.write(f'Description de {colonne_selectionnee}')
    if colonne_selectionnee =='year':
      category_counts = dfj['year'].value_counts()
      st.write("Nombre d'enregistrements par année")
      st.write(category_counts)
    else : 
      st.write(f'Description de {colonne_selectionnee}')
      st.write(df[colonne_selectionnee].describe())
    
  
    sns.set(style="whitegrid")
    
         #graphique de densité kernel (KDE)
    st.write("<b><u>Analyse du bonheur - approche par régions</u></b><br>",unsafe_allow_html=True)
    st.write("<b><u>Graphique de densité Kernel (KDE)</u></b><br>",unsafe_allow_html=True)

    plt.figure(figsize=(15,8))
    sns.kdeplot(x=df4["Life Ladder"], hue=df4["Regional indicator"],fill=True ,linewidth=2)
    plt.axvline(df4["Life Ladder"].mean(),c="black")
    plt.title("Graphique de densité Kernel (KDE)")
    st.pyplot(plt)
        

    # Sélectionner uniquement les colonnes numériques du DataFrame
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64'])

    st.write("<b><u>Matrice de corrélation</u></b><br>",unsafe_allow_html=True)

    # Calculer la matrice de corrélation pour les colonnes numériques
    cor = round(colonnes_numeriques.corr(), 2)
    # Créer un heatmap à l'aide de seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
    plt.title("Correlations avec Life Ladder")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(plt)

    st.write("<b><u>Relations entre variables</u></b><br>",unsafe_allow_html=True)

    fig = plt.figure(figsize = (20,8))
    fig.suptitle("Relation entre nos variables les plus fortement corrélées avec la variable cible")

    plt.subplot(131)
    plt.scatter('Log GDP per capita','Life Ladder', data = df4)
    plt.xlabel('Log GDP per capita')

    plt.subplot(132)
    plt.scatter('Social support','Life Ladder', data = df4,c='r')
    plt.xlabel('Social support')

    plt.subplot(133)
    plt.scatter('Healthy life expectancy at birth','Life Ladder', data = df4,c='g')
    plt.xlabel('Healthy life expectancy at birth');
    st.pyplot(plt)
    import plotly.express as px
    # Sélectionner les colonnes nécessaires du DataFrame
    selected_columns = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']
    df_selected = df4[selected_columns]

    # Créer un scatter plot interactif avec Plotly Express
    fig = px.scatter_matrix(df_selected, dimensions=selected_columns, labels=selected_columns)

    # Personnaliser le titre du graphique
    fig.update_layout(
      title='Scatter Plot Matrix',
      title_x=0.5,
      title_font=dict(size=20)
    )

    # Afficher le graphique interactif dans Streamlit
    st.plotly_chart(fig)

    #plt.figure(figsize=(10, 6))
    #p = sns.barplot(y=df5['Country name'].head(10), x=df5['Life Ladder'].head(10))
    #p.set_title("Top 10 des pays les plus heureux")
    # Afficher le graphique dans Streamlit
    #st.pyplot(plt)
    #plt.figure(figsize=(10, 6))
    #p= sns.barplot(y=df5['Country name'].tail(10), x=df5['Life Ladder'].tail(10))
    #p.set_title("Top 10 des pays les plus malheureux");
    #st.pyplot(plt)
    st.write("<b><u>Variable Generosity</u></b><br>",unsafe_allow_html=True)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Focus sur la variable Generosity")
    
    ax[0].set_title('Sa représentation par région')
    sns.barplot(x='Regional indicator', y='Generosity', data=df4, ax=ax[0])
    ax[0].tick_params(axis='x', rotation=45)
    
    ax[1].set_title('Sa représentation par année')
    sns.barplot(x='year', y='Generosity', data=df4, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    
    # Supprimez le subplot vide
    ax[2].axis('off')
    
    # Affichez le graphique dans Streamlit
    st.pyplot(fig)
    
    
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
                #st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
                st.write("<b><u>Modélisation</u></b>",unsafe_allow_html=True)
                #st.write(f"MAE: {mae}")
                #st.write(f"RMSE: {rmse}")
                #st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae2 = mean_absolute_error(y_pred_2021, y_2021)
                rmse2 = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                st.write(f"<table><tr><th>Target Encoder</th><th>MAE</th><th>RMSE</th></tr><tr><td>Jeu d'entrainement</td><td>{round(mae,4)}</td><td>{round(rmse,4)}</td></tr><tr><td>Prédictions 2021</td><td>{round(mae2,4)}</td><td>{round(rmse2,4)}</td></tr></table>",unsafe_allow_html=True)
#                st.write(f"MAE: {mae}")
#                st.write(f"RMSE: {rmse}")
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
                
                if model_name == 'Random Forest':
                  st.write("<b><u>Features importance</u></b>",unsafe_allow_html=True)
                  st.write("<ul><li>Variable la plus utilisée : Country name</li><li>Variables les moins utilisées : Healthy et Perceptions of corruption</li><li>Les autres variables sont assez équilibrées</li></ul>",unsafe_allow_html=True)
                
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
   #             st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
                st.write("<b><u>Modélisation</u></b>",unsafe_allow_html=True)
    #            st.write(f"MAE: {mae}")
     #           st.write(f"RMSE: {rmse}")
      #          st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021l)
                # Calcul des métriques
                mae2 = mean_absolute_error(y_pred_2021, y_2021)
                rmse2 = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
 #               st.write(f"MAE: {mae}")
  #              st.write(f"RMSE: {rmse}")
                #st.write(f"RMSE_REVU: {RMSE_REVU}")
                st.write(f"<table><tr><th>Target Encoder</th><th>MAE</th><th>RMSE</th></tr><tr><td>Jeu d'entrainement</td><td>{round(mae,4)}</td><td>{round(rmse,4)}</td></tr><tr><td>Prédictions 2021</td><td>{round(mae2,4)}</td><td>{round(rmse2,4)}</td></tr></table>",unsafe_allow_html=True)
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
                svr2.fit(X_trains, y_trains)
                y_preds2 = svr2.predict(X_tests)
                X_2021s = gestion_nan1(X_2021)
                # Calcul des métriques
                mae = mean_absolute_error(y_tests, y_preds)
                rmse = mean_squared_error(y_tests, y_preds, squared=False)
                mae2 = mean_absolute_error(y_tests, y_preds2)
                rmse2 = mean_squared_error(y_tests, y_preds2, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write("<b><u>Modélisation</b></u>",unsafe_allow_html=True)
#                st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
 #               st.write(f"MAE: {mae}")
  #              st.write(f"RMSE: {rmse}")
   #             st.write("<b><u>Modélisation jeu d'entraînement One Hot</u></b>",unsafe_allow_html=True)
    #            st.write(f"MAE: {mae2}")
     #           st.write(f"RMSE: {rmse2}")
      #          st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021s)
                y_pred_20212 = svr2.predict(X_2021s)
                # Calcul des métriques
                mae3 = mean_absolute_error(y_pred_2021, y_2021)
                rmse3 = mean_squared_error(y_pred_2021, y_2021, squared=False)
                mae4 = mean_absolute_error(y_pred_20212, y_2021)
                rmse4 = mean_squared_error(y_pred_20212, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
       #         st.write(f"MAE: {mae3}")
        #        st.write(f"RMSE: {rmse3}")
         #       st.write("<b><u>Prédictions 2021 One Hot</u></b>",unsafe_allow_html=True)
          #      st.write(f"MAE: {mae4}")
           #     st.write(f"RMSE: {rmse4}")
                #st.write(f"RMSE_REVU: {RMSE_REVU}")
                st.write(f"<table><tr><th>Target Encoder</th><th>MAE</th><th>RMSE</th></tr><tr><td>Jeu d'entrainement</td><td>{round(mae,4)}</td><td>{round(rmse,4)}</td></tr><tr><td>Prédictions 2021</td><td>{round(mae2,4)}</td><td>{round(rmse2,4)}</td></tr></table>",unsafe_allow_html=True)
                st.write(f"<br><table><tr><th>One Hot Encoder</th><th>MAE</th><th>RMSE</th></tr><tr><td>Jeu d'entrainement</td><td>{round(mae3,4)}</td><td>{round(rmse3,4)}</td></tr><tr><td>Prédictions 2021</td><td>{round(mae4,4)}</td><td>{round(rmse4,4)}</td></tr></table>",unsafe_allow_html=True)
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
                st.write("<b><u>Features importance</u></b>",unsafe_allow_html=True)
                st.write("<ul><li>Les indicateurs socio économiques sont privilégiés</li><li>Seul l'indicateur Healthy a été relayé au second plan</li><li>Les variables Generosity et Perceptions of corruption étaient les moins corrélées (en accord avec notre exloration)</li></ul>",unsafe_allow_html=True)
                
            elif model_name =='KNN':
                st.image(pknn)
                st.write("<b><u>Optimisation</b></u>",unsafe_allow_html=True)
                st.image(oknn)
                model.fit(X_traink, y_traink)
                y_predk = model.predict(X_testk)
                knn2.fit(X_traink, y_traink)
                y_predk2 = knn2.predict(X_testk)
                # Calcul des métriques
                mae = mean_absolute_error(y_testk, y_predk)
                rmse = mean_squared_error(y_testk, y_predk, squared=False)
                mae2 = mean_absolute_error(y_testk, y_predk2)
                rmse2 = mean_squared_error(y_testk, y_predk2, squared=False)
                # Affichage des résultats
                #st.write(f"Modèle: {model_name}")
                st.write("<b><u>Modélisation</b></u>",unsafe_allow_html=True)
  #              st.write("<b><u>Modélisation jeu d'entraînement</u></b>",unsafe_allow_html=True)
   #             st.write(f"MAE: {mae}")
    #            st.write(f"RMSE: {rmse}")
     #           st.write("<b><u>Modélisation jeu d'entraînement One Hot</u></b>",unsafe_allow_html=True)
      #          st.write(f"MAE: {mae2}")
       #         st.write(f"RMSE: {rmse2}")
        #        st.write("<b><u>Prédictions 2021</u></b>",unsafe_allow_html=True)
                y_pred_2021 = model.predict(X_2021)
                # Calcul des métriques
                mae3 = mean_absolute_error(y_pred_2021, y_2021)
                rmse3 = mean_squared_error(y_pred_2021, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)
                
 #               st.write(f"MAE: {mae3}")
 #               st.write(f"RMSE: {rmse3}")
     #           st.write("<b><u>Prédictions 2021 One Hot</u></b>",unsafe_allow_html=True)
                y_pred_20212 = knn2.predict(X_2021)
                # Calcul des métriques
                mae4 = mean_absolute_error(y_pred_20212, y_2021)
                rmse4 = mean_squared_error(y_pred_20212, y_2021, squared=False)
                #RMSE_REVU = rmse_revu(y_pred_2021, y_2021)                
                st.write(f"<table><tr><th>Target Encoder</th><th>MAE</th><th>RMSE</th></tr><tr><td>Jeu d'entrainement</td><td>{round(mae,4)}</td><td>{round(rmse,4)}</td></tr><tr><td>Prédictions 2021</td><td>{round(mae2,4)}</td><td>{round(rmse2,4)}</td></tr></table>",unsafe_allow_html=True)
                st.write(f"<br><table><tr><th>One Hot Encoder</th><th>MAE</th><th>RMSE</th></tr><tr><td>Jeu d'entrainement</td><td>{round(mae3,4)}</td><td>{round(rmse3,4)}</td></tr><tr><td>Prédictions 2021</td><td>{round(mae4,4)}</td><td>{round(rmse4,4)}</td></tr></table>",unsafe_allow_html=True)

                #                st.write(f"MAE: {mae4}")
#                st.write(f"RMSE: {rmse4}")
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
                st.write("<b><u>Features importance</u></b>",unsafe_allow_html=True)
                st.write("<ul><li>Les indicateurs privilégiés sont : Log GDP, Country, Healthy</li><li>Les moins utilisés sont : Generosity et Positive affect</li></ul>",unsafe_allow_html=True)
                
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

#        # Dictionnaire pour stocker les valeurs saisies par l'utilisateur
#        user_inputs = {}
        # Initialisation du dictionnaire user_inputs dans st.session_state si ce n'est pas déjà fait
        if "user_inputs" not in st.session_state:
          st.session_state.user_inputs = {}

        # Dans le formulaire, utilisez le dictionnaire user_inputs de st.session_state
        for column in list(df_ensemble.columns)[2:-1]:
          #if column not in st.session_state.user_inputs:
          st.session_state.user_inputs[column] = st.text_input(column)

        # Zones de texte pour les colonnes du DataFrame df_ensemble à partir de la deuxième colonne
        #for column in list(df_ensemble.columns)[2:-1]:
        #    user_inputs[column] = st.text_input(column)

        # Bouton "Entraîner les modèles"
        submit_button = st.form_submit_button('Prédiction Life Ladder')

    # Création du DataFrame à partir des valeurs saisies par l'utilisateur
    if submit_button:
        selected_model = st.session_state.model_select
        # Filtrer les valeurs non vides
        #user_inputs = {key: value for key, value in user_inputs.items() if value != ''}
        #user_inputs = {clef: float(valeur.replace(',', '.')) for clef, valeur in user_inputs.items()}
#        # Dictionnaire pour stocker les valeurs saisies par l'utilisateur
#        user_inputs = {}

 #       # Zones de texte pour les colonnes du DataFrame df_ensemble à partir de la deuxième colonne
 #       for column in list(df_ensemble.columns)[2:-1]:
 #         user_inputs[column] = st.text_input(column)

  #      # Remplacer les champs vides par np.nan
   #     user_inputs = {key: float(value.replace(',', '.')) if value != '' else np.nan for key, value in user_inputs.items()}
        # Remplacer les champs vides par np.nan dans st.session_state.user_inputs
        user_inputs = {
          key: float(value.replace(',', '.')) if value != '' else np.nan 
          for key, value in st.session_state.user_inputs.items()
        }
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
                    X_train_new = gestion_nan2(X_train_new, X_trains)
                    y_pred_saisie = model.predict(X_train_new)
                    #st.write(y_pred_saisie)
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
