import numpy as np
import pandas as pd
from load_data import load_all
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

def map_attributes(data):
    gender_map = {'M':0, 'F':1}
    occupation_map = {'administrator':0, 'artist':1, 'doctor':2, 'educator':3, 'engineer':4, 'entertainment':5,
            'executive':6, 'healthcare':7, 'homemaker':8, 'lawyer':9, 'librarian':10, 'marketing':11,
            'none':12, 'other':13, 'programmer':14, 'retired':15, 'salesman':16, 'scientist':17, 'student':18,
            'technician':19, 'writer':20}
    data["gender"] = data["gender"].map(gender_map)
    data["occupation"] = data["occupation"].map(occupation_map)
    return data


rating, users, movies = load_all("Recommender System/")
data = pd.merge(rating, users, on="user_id")
data = map_attributes(data)
data = pd.merge(data, movies, on="item_id").fillna(0)

features = ["age", "gender", "occupation","release_date", "unknown", "Action", "Adventure", "Animation","Children's", "Comedy", "Crime",
         "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

x_train, x_test, y_train, y_test = train_test_split(data[features], data[["rating"]], test_size=0.2, random_state=1)

pgrid = ParameterGrid({
    'criterion': ['entropy', 'gini'], 
    'splitter': ['best', 'random'], 
    'max_depth': [5, 7, 10], 
    'min_samples_leaf': [1, 3, 5], 
    'min_samples_split': [2, 3, 4], 
    'max_features': [7, 8, 9], 
    'random_state': [123]
    })

acc = 0
best_p = {}
for p in pgrid:
    dt = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=9)), ('dt', DecisionTreeClassifier(**p))])
    dt.fit(x_train, y_train)
    s = dt.score(x_test, y_test)
    if s > acc:
        acc = s
        best_p = p

if best_p != {}:
    print('best parameter :', best_p)
    dt = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=9)), ('dt', DecisionTreeClassifier(**best_p))])
    dt.fit(x_train, y_train)
    print('train accuracy :', dt.score(x_train, y_train))
    cvs = cross_val_score(dt, data[features], data[['rating']], cv=5)
    print(cvs)
    print('mean :', np.mean(cvs))