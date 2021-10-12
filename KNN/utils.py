import numpy as np

from scipy.spatial import distance
from functools import reduce

def dist(sample1, sample2, n=2):
    return distance.minkowski(sample1, sample2, n)

def max_distance(dataset, n=2):
    mx = 0
    for x in dataset:
        for y in dataset:
            mx = max(mx, dist(x, y, n))
    return mx

def wrong_indexes(predicts, truth):
    wrong_ind =[]
    for index in range(len(predicts)):
        if predicts[index] != truth[index]:
            wrong_ind.append(index)
    return wrong_ind
    
def accuracy(predicts, ground_truth):
    truth = reduce(lambda x, elem: x + 1 if elem[0] == elem[1] else x, zip(predicts, ground_truth), 0)
    print(str(truth) + "/" + str(len(predicts)))
    print("Accuracy: " + str(truth/len(predicts)))