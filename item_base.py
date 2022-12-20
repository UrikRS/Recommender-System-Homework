import numpy as np
from math import sqrt
from load_data import load_train_test
from sklearn.neighbors import NearestNeighbors

class Item_Base_CF:
    def __init__(self, n_neighbors=3) -> None:
        self.item_user = np.empty((1682, 943))
        self.n = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=self.n, metric="cosine")
        self.distances, self.indices = np.empty((1682, self.n)), np.empty((1682, self.n))
        self.similarities = np.empty((1682, self.n))

    def fit(self, data):
        self.item_user = data.pivot(index="item_id", columns="user_id", values="rating")
        # Fail to adjust item-to-user matrix with mean of user-ratings
        # self.item_user = self.item_user.subtract(self.item_user.mean(axis=1), axis = 0).fillna(0)
        self.item_user = self.item_user.reindex(index=np.arange(1,1683), fill_value=0).fillna(0)
        # Use sklearn NearestNeighbors to find n similar item's indices & cosine similarities
        self.knn.fit(self.item_user)
        self.distances, self.indices = self.knn.kneighbors(n_neighbors=self.n)
        self.similarities = 1 - self.distances

    def predict(self, item_id, user_id, epsilon=1e-8):
        # Function to predict rating via item_id & user_id
        # Using formula in 01_Neighborhood-based_collaborative_filtering.pptx page 32
        pred_r = self.item_user.iloc[item_id-1,user_id-1]
        sim, ind = self.similarities[item_id-1], self.indices[item_id-1]
        product = 1
        product_sum = 0
        if self.item_user.iloc[item_id-1,user_id-1]!=0:
            return pred_r
        else:
            for i in range(len(ind)):
                product = self.item_user.iloc[ind[i],user_id-1] * sim[i]
                product_sum += product
            pred_r = product_sum / (np.sum(sim)+epsilon)
        return pred_r

    def RMSE(self, data):
        # Function to calculate RMSE of predicted rating & actual rating
        x = data[["item_id", "user_id"]].to_numpy()
        y = data[["rating"]].to_numpy()
        losses = []
        for i in range(len(x)):
            losses.append((self.predict(x[i][0], x[i][1]) - y[i]).item())
        return sqrt(np.nanmean(np.square(losses)))

if __name__ == '__main__':
    n_neighbors = [2, 4, 8]
    for n in n_neighbors:
        print('-'*10+str(n)+'-neighbors'+'-'*10)
        # 5-fold
        for i in range(1, 6):
            train, test = load_train_test("Recommender System/", i)
            cf = Item_Base_CF(n)
            cf.fit(train)
            print('u'+str(i)+'.test RMSE :', cf.RMSE(test))