# import numpy as np
from numpy import array as np_array, sqrt as np_sqrt, sum as np_num
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error

def run_knn(X_train, X_test, y_train, y_test, k=3):
    knn = KNN(k)
    knn.set_data(X_train, y_train)
    cat = knn.predict_set(X_test)
    knn.evaluate(y_test, cat)

class KNN():
    '''
    An implementation of the k-nearest neighbours algorithm.

    metric: euclidean distance
    default number of neighbours: 3
    
    The results could be improved by experimenting with different metrics and different k values
    '''
    def __init__(self, k=3) -> None:
        self.k = k
        self.metric = self.euclidean_distance
        self.X, self.y = None, None
    
    def set_data(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, point):
        distances = [(i, self.metric(self.X[i], point)) for i in range(len(self.X))]
        distances.sort()
        distances = sorted(distances, key = lambda x: x[1], reverse = True)
        nn = [self.y[i] for i,j in distances[:self.k]]
        winner = max(set(nn), key=nn.count)
        return winner
    
    def euclidean_distance(self, x1, x2):
        dist = np_num([(x1[i] - x2[i])**2 for i in range(len(x1))])
        return sqrt(dist)
    
    def predict_set(self, X_test):
        prediction = np_array([self.predict(x) for x in X_test])
        return prediction
    
    def evaluate(self, y_test, y_predict):
        print("R2 score : ", r2_score(y_test, y_predict))
        print("Mean Absolute Error :", mean_absolute_error(y_test,y_predict))
        print("Root Mean Squared Error:", np_sqrt(mean_squared_error(y_test, y_predict)))
