from scipy.spatial import distance
import numpy as np

train_data = 0

def dist(sample1,sample2,n=2):
    return distance.minkowski(sample1,sample2,n)

def max_distance(n=2):
    mx = 0
    for x in train_data:
        for y in train_data:
            mx = max(mx, dist(x,y,n))
    return mx

def non_zero_weigth(weights):
    nzarray = []
    for index in range(len(weights)):
        if (weights[index]!= 0):
            nzarray.append(index)
    return np.array(nzarray)