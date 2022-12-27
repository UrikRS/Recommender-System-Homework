import pandas as pd
from load_data import load_all
import matplotlib.pyplot as plt

# Load csv to dataframe
rating, users, movies = load_all("Recommender System/")

# View some data
print(rating.head(5))
print(users.head(5))
print(movies.head(5))

# Make user age distribution plot
users.age.plot.hist(bins=25)
plt.title("User's Ages Distribution")
plt.ylabel("Number of Users")
plt.xlabel("Age")
plt.savefig("Recommender System/Homework/Pic/user_age.png")
plt.close()

# Make user occupation distribution plot
occupation_count = users[["user_id", "occupation"]].groupby("occupation", as_index=False).size()
plt.pie(occupation_count["size"], labels=occupation_count["occupation"])
plt.title("User's Occupation Distribution")
plt.axis("equal")
plt.savefig("Recommender System/Homework/Pic/user_occupation.png")
plt.close()

# Make user gender percentage plot
gender_count = users[["user_id", "gender"]].groupby("gender", as_index=False).size()
plt.pie(gender_count["size"], labels=gender_count["gender"], autopct='%1.0f%%')
plt.title("User's Gender Distribution")
plt.axis("equal")
plt.savefig("Recommender System/Homework/Pic/user_gender.png")
plt.close()

# Make item release year distribution plot
movies[["release_date"]].groupby("release_date").size().plot(kind='area')
plt.title("Movie's Release Year Distribution")
plt.ylabel("Number of Movies")
plt.xlabel("Release Year")
plt.savefig("Recommender System/Homework/Pic/movies_release_year.png")
plt.close()

# Make item genre distribution plot
genre_count = pd.DataFrame({"genre":["unknown", "Action", "Adventure", "Animation", "Children's",
         "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
         "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], 
         "count": [movies["unknown"].sum(), movies["Action"].sum(), movies["Adventure"].sum(),
         movies["Animation"].sum(), movies["Children's"].sum(), movies["Comedy"].sum(),
         movies["Crime"].sum(), movies["Documentary"].sum(), movies["Drama"].sum(), 
         movies["Fantasy"].sum(), movies["Film-Noir"].sum(), movies["Horror"].sum(), 
         movies["Musical"].sum(), movies["Mystery"].sum(), movies["Romance"].sum(), 
         movies["Sci-Fi"].sum(), movies["Thriller"].sum(), movies["War"].sum(), movies["Western"].sum()]})
plt.pie(genre_count["count"], labels=genre_count["genre"])
plt.title("Movie's Genre Distribution")
plt.axis("equal")
plt.savefig("Recommender System/Homework/Pic/movies_genre.png")
plt.close()
