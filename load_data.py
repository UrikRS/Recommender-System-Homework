import pandas as pd

# Set dataframes headers
rating_header = ["user_id", "item_id", "rating", "timestamp"]
user_header = ["user_id", "age", "gender", "occupation", "zip_code"]
movie_header = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
         "unknown", "Action", "Adventure", "Animation","Children's", "Comedy", "Crime",
         "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
         "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Function of loading u.data, u.user, u.item to dataframe
def load_all(directory):
    rating = pd.read_csv(directory+"ml-100k/u.data", sep='\t', header=None, names=rating_header)
    users = pd.read_csv(directory+"ml-100k/u.user", sep='|', header=None, names=user_header)
    movies = pd.read_csv(directory+"ml-100k/u.item", sep='|', header=None, encoding = 'latin1', names=movie_header)
    # Drop useless informations
    rating = rating.drop(columns="timestamp")
    users = users.drop(columns="zip_code")
    movies = movies.drop(columns=["IMDb_URL", "video_release_date"])
    # Modify movie's release date to release year
    movies["release_date"] = pd.to_datetime(movies["release_date"]).dt.strftime('%Y')
    return rating, users, movies

# Function of loading u.data to dataframe
def load_rating(directory):
    rating = pd.read_csv(directory+"ml-100k/u.data", sep='\t', header=None, names=rating_header)
    # Drop useless informations
    rating = rating.drop(columns="timestamp")
    return rating

def load_train_test(directory, i):
    train = pd.read_csv(directory+"ml-100k/u"+str(i)+".base", sep='\t', header=None, names=rating_header)
    test = pd.read_csv(directory+"ml-100k/u"+str(i)+".test", sep='\t', header=None, names=rating_header)
    # Drop useless informations
    train = train.drop(columns="timestamp")
    test = test.drop(columns="timestamp")
    return train, test