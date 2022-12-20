import numpy as np
import pandas as pd
from load_data import load_all
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def map_str_to_number(data):
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
rating_user = map_str_to_number(rating_user)
x_train, x_test, y_train, y_test = train_test_split(rating_user[["user_id", "age", "gender", "occupation", "item_id"]], rating[["rating"]], test_size=0.2, random_state=1)
nb = GaussianNB()
nb.fit(x_train.to_numpy(), y_train.to_numpy().ravel())
print(nb.score(x_test.to_numpy(), y_test.to_numpy().ravel()))