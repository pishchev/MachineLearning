import kernels
import numpy as np
import utils

class Trainer:
    def __init__(self):
        pass
    def train(self, model, train_X, train_Y, epoch):
        model.weights[0] = 1
        for repeat in range(epoch):
            copy_weights = model.weights.copy()
            for index in range(len(train_X)):
                sample = train_X[index]
                label = train_Y[index]
                if (model.predict(sample, train_X, train_Y) != label):
                    model.weights[index] += 1
            if (np.array_equal(copy_weights, model.weights)):
                print("Ready!")
                break
            print("Epoch " + str(repeat + 1) +" completed")