import pandas as pd
from load_data import load_all
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

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
rating_user = pd.merge(rating, users, on="user_id")
rating_user = map_attributes(rating_user)
data = pd.merge(rating_user, movies, on="item_id").fillna(0)

features = ["age", "gender", "occupation","release_date", "unknown", "Action", "Adventure", "Animation","Children's", "Comedy", "Crime",
         "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

x_train, x_test, y_train, y_test = train_test_split(data[features], data[["rating"]], test_size=0.2, random_state=1)
dt = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=5)), ('dt', DecisionTreeRegressor())])
dt.fit(x_train, y_train.to_numpy().ravel())

cvs = cross_val_score(dt, data[features].fillna(0), data[["rating"]].to_numpy().ravel(), cv=5)
print(cvs)