import kernels
import numpy as np
import utils

train_data = 0
train_labels = 0
test_data = 0
test_labels = 0

h = 0
n = 0

def SetData(itrain_data, itest_data, itrain_labels,itest_labels):
    global train_data, train_labels, test_data, test_labels

    train_data = itrain_data
    train_labels = itrain_labels
    test_data = itest_data
    test_labels = itest_labels

def predict(sample, weights, kernel=kernels.Gauss):
    resVec = np.zeros(train_labels.shape[0])
    for index in range(len(train_data)):
        label = train_labels[index]
        train_sample = train_data[index]
        weight = weights[index]
        resVec[label] +=  weight * kernel(utils.dist(train_sample,sample,n) / h)
    return np.argmax(resVec)  

def train_weights(times=20, kernel=kernels.Gauss):
    weights = np.zeros_like(train_labels)
    weights[0] = 1

    for repeat in range(times):
        copy_weights = weights.copy()
        for index in range(len(train_data)):
            sample = train_data[index]
            label = train_labels[index]
            if (predict(sample, weights, kernel) != label):
                weights[index] += 1
        if (np.array_equal(copy_weights, weights)):
            print("Ready!")
            break
        print("Epoch " + str(repeat + 1) +" completed")

    return weights

def accuracy(weights, kernel=kernels.Gauss):
    trues = 0
    wrongs = []
    for index in range(len(test_labels)):
        sample = test_data[index]
        label = test_labels[index]
        if (predict(sample, weights, kernel) == label):
            trues += 1
        else:
            wrongs.append(index)
    print(str(trues)+"/"+str(len(test_labels)))
    return trues/len(test_labels), wrongs