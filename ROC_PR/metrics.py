from functools import reduce

def TP(arr,treshold):
    return reduce(lambda x,y: x+1 if y[0] >= treshold and y[1] == 0 else x, arr, 0 )

def TN(arr,treshold):
    return reduce(lambda x,y: x+1 if y[0] < treshold and y[1] == 1 else x,  arr,0 )

def FP(arr,treshold):
    return reduce(lambda x,y: x+1 if y[0] >= treshold and y[1] == 1 else x, arr, 0 )

def FN(arr,treshold):
    return reduce(lambda x,y: x+1 if y[0] < treshold and y[1] == 0 else x, arr, 0 )

def recall(probe_predicts, treshold):
    return TP(probe_predicts,treshold)/(TP(probe_predicts,treshold)+FN(probe_predicts,treshold))

def precision(probe_predicts, treshold):
    return TP(probe_predicts,treshold)/(TP(probe_predicts,treshold)+FP(probe_predicts,treshold))

def TPR(probe_predicts, treshold):
    return recall(probe_predicts, treshold)

def FPR(probe_predicts, treshold):
    return FP(probe_predicts,treshold)/(FP(probe_predicts,treshold)+TN(probe_predicts,treshold))