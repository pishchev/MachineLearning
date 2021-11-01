import numpy as np
import kernels

import utils

class Model:
    def __init__(self, weights_size, max_distance, kernel=kernels.Gauss, pow_distance=2):
        self.weights = np.zeros(weights_size, int)
        self.pow_distance = pow_distance
        self.kernel = kernel
        self.max_distance = max_distance
        
    def non_zero_weights(self, is_zero):
        array = []
        for index in range(len(self.weights)):
            if (not is_zero and self.weights[index] != 0):
                array.append(index)
            elif (is_zero and self.weights[index] == 0):
                array.append(index)

        return np.array(array)

    def predicts(self, test_X, train_X, train_Y):
        preds = list(map(lambda x: self.predict(x, train_X, train_Y), test_X))
        return np.array(preds)

    def predict(self, sample, train_X, train_Y):
        resVec = np.zeros(train_Y.shape[0])

        diffs = np.array(list(map(lambda x: self.kernel(utils.dist(x, sample, self.pow_distance) / self.max_distance), train_X)))
        weights = self.weights * diffs
        nresVec = np.zeros(train_Y.shape[0])
        for i in range(len(train_Y)):
            nresVec[i] = np.sum( weights[ i == train_Y] )

        return np.argmax(nresVec)
