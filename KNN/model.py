import numpy as np
import kernels

import utils

class Model:
    def __init__(self, weights_size, max_distance, kernel=kernels.Gauss, pow_distance=2):
        self.weights = np.zeros(weights_size, int)
        self.pow_distance = pow_distance
        self.kernel = kernel
        self.max_distance = max_distance
        
    def non_zero_weights(self):
        nzarray = []
        for index in range(len(self.weights)):
            if (self.weights[index]!= 0):
                nzarray.append(index)
        return np.array(nzarray)

    def predicts(self, test_X, train_X, train_Y):
        preds = list(map(lambda x: self.predict(x, train_X, train_Y) , test_X))
        return np.array(preds)

    def predict(self, sample, train_X, train_Y):
        resVec = np.zeros(train_Y.shape[0])
        for index in range(len(train_X)):
            label = train_Y[index]
            train_sample = train_X[index]
            weight = self.weights[index]
            resVec[label] +=  weight * self.kernel(utils.dist(train_sample,sample,self.pow_distance) / self.max_distance)
        return np.argmax(resVec)
