from scipy.spatial import distance
import numpy as np

def dist(sample1,sample2,n=2):
    return distance.minkowski(sample1,sample2,n)

def max_distance(dataset, n=2):
    mx = 0
    for x in dataset:
        for y in dataset:
            mx = max(mx, dist(x,y,n))
    return mx

def wrong_indexes(predicts, truth):
    wrong_ind =[]
    for index in range(len(predicts)):
        if predicts[index] != truth[index]:
            wrong_ind.append(index)
    return wrong_ind
    
def accuracy(predicts, ground_truth):
    truth = 0
    for index in range(len(predicts)):
        if predicts[index] == ground_truth[index]:
            truth +=1 
    print(str(truth) +"/" + str(len(predicts)))
    print("Accuracy: "+str(truth/len(predicts)))