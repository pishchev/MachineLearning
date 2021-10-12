#без циклов

def TP(arr,treshold):
    sum = 0
    for i in arr:
        sum += 1 if i[0] >= treshold and i[1] == 0 else 0
    return sum

def TN(arr,treshold):
    sum = 0
    for i in arr:
        sum += 1 if i[0] < treshold and i[1] == 1 else 0
    return sum

def FP(arr,treshold):
    sum = 0
    for i in arr:
        sum += 1 if i[0] >= treshold and i[1] == 1 else 0
    return sum

def FN(arr,treshold):
    sum = 0
    for i in arr:
        sum += 1 if i[0] < treshold and i[1] == 0 else 0
    return sum

def recall(probe_predicts, treshold):
    return TP(probe_predicts,treshold)/(TP(probe_predicts,treshold)+FN(probe_predicts,treshold))

def precision(probe_predicts, treshold):
    return TP(probe_predicts,treshold)/(TP(probe_predicts,treshold)+FP(probe_predicts,treshold))

def TPR(probe_predicts, treshold):
    return recall(probe_predicts, treshold)

def FPR(probe_predicts, treshold):
    return FP(probe_predicts,treshold)/(FP(probe_predicts,treshold)+TN(probe_predicts,treshold))