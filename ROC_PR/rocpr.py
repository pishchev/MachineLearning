import metrics
import utils
import matplotlib.pyplot as plt

def ROCSTEP(probe_predicts):
    points = []
    curx = 0
    cury = 0
    area = 0

    neg_step = 1/utils.calcVals(probe_predicts,1)
    pos_step = 1/utils.calcVals(probe_predicts,0)

    points.append([curx,cury])
    for i in probe_predicts:
        if (i[1] == 0):
            cury += pos_step
        else:
            curx += neg_step
            area += cury * neg_step
        points.append([curx,cury])

    print('Area ROCSTEP = ' + str(area))
    plt.title("ROC (Step method)")
    plt.plot(*zip(*points))
    plt.show()

def ROC(probe_predicts):
    points = [[0,0]]
    for i in range(len(probe_predicts)):
        points.append([metrics.FPR(probe_predicts, probe_predicts[i][0]), metrics.TPR(probe_predicts, probe_predicts[i][0])])
   
    plt.title("ROC")
    print('Area ROC= ' + str(utils.areaCalc(points)))
    plt.plot(*zip(*points))
    plt.show()
    
def PR(probe_predicts):
    points = [[0,1]]
    for i in range(len(probe_predicts)):
        points.append([metrics.recall(probe_predicts, probe_predicts[i][0]), metrics.precision(probe_predicts, probe_predicts[i][0])])

    plt.title("PR")
    print('Area = ' + str(utils.areaCalc(points)))
    plt.plot(*zip(*points))
    plt.show()