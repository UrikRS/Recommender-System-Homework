from sklearn.cluster import KMeans
# Clustering the movies by genre

genre = pd.read_csv("ml-100k/u.genre", sep = '|', header = None)
genre_list = genre[0].values
movie_set_genre = movies[genre_list]

k = 50
kmeans = KMeans(n_clusters=k)
kmeans.fit_predict(movie_set_genre)

#print(len(kmeans.labels_))


new_data = pd.DataFrame()
new_data['item_id'] = movies["item_id"]
new_data['labels'] = kmeans.labels_

A = new_data['item_id'].tolist()
B = new_data['labels'].tolist()



x_train = pd.read_csv("ml-100k/ub.base", sep='\t', header=None, names=rating_header)
y_train = pd.read_csv("ml-100k/ub.base", sep='\t', header=None, names=rating_header)
y_train = y_train['rating']
x_train = x_train.drop(columns="rating")
x_train = x_train.drop(columns="timestamp")
new_x_train = x_train


for a in range(0,len(A)):
    new_x_train['item_id'].replace(A[a],B[a],inplace=True)


x_test = pd.read_csv("ml-100k/ub.test", sep='\t', header=None, names=rating_header)
y_test = pd.read_csv("ml-100k/ub.test", sep='\t', header=None, names=rating_header)
y_test = y_test['rating']
x_test = x_test.drop(columns="rating")
x_test = x_test.drop(columns="timestamp")
new_x_test = x_test

for a in range(0,len(A)):
    new_x_test['item_id'].replace(A[a],B[a],inplace=True)


from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

clf=svm.SVC(kernel='rbf',C=3,gamma='auto')
clf.fit(x_train,y_train)

clf2=svm.SVC(kernel='rbf',C=3,gamma='auto')
clf2.fit(new_x_train,y_train)

clf.predict(x_test)
clf2.predict(new_x_test)

print(clf.score(x_train,y_train))
print(clf.score(x_test, y_test))

print(clf2.score(new_x_train,y_train))
print(clf2.score(new_x_test, y_test))from sklearn.cluster import KMeans
# Clustering the movies by genre

genre = pd.read_csv("ml-100k/u.genre", sep = '|', header = None)
genre_list = genre[0].values
movie_set_genre = movies[genre_list]

k = 50
kmeans = KMeans(n_clusters=k)
kmeans.fit_predict(movie_set_genre)

#print(len(kmeans.labels_))


new_data = pd.DataFrame()
new_data['item_id'] = movies["item_id"]
new_data['labels'] = kmeans.labels_

A = new_data['item_id'].tolist()
B = new_data['labels'].tolist()



x_train = pd.read_csv("ml-100k/ub.base", sep='\t', header=None, names=rating_header)
y_train = pd.read_csv("ml-100k/ub.base", sep='\t', header=None, names=rating_header)
y_train = y_train['rating']
x_train = x_train.drop(columns="rating")
x_train = x_train.drop(columns="timestamp")
new_x_train = x_train


for a in range(0,len(A)):
    new_x_train['item_id'].replace(A[a],B[a],inplace=True)


x_test = pd.read_csv("ml-100k/ub.test", sep='\t', header=None, names=rating_header)
y_test = pd.read_csv("ml-100k/ub.test", sep='\t', header=None, names=rating_header)
y_test = y_test['rating']
x_test = x_test.drop(columns="rating")
x_test = x_test.drop(columns="timestamp")
new_x_test = x_test

for a in range(0,len(A)):
    new_x_test['item_id'].replace(A[a],B[a],inplace=True)


from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

clf=svm.SVC(kernel='rbf',C=3,gamma='auto')
clf.fit(x_train,y_train)

clf2=svm.SVC(kernel='rbf',C=3,gamma='auto')
clf2.fit(new_x_train,y_train)

clf.predict(x_test)
clf2.predict(new_x_test)

print(clf.score(x_train,y_train))
print(clf.score(x_test, y_test))

print(clf2.score(new_x_train,y_train))
print(clf2.score(new_x_test, y_test))
