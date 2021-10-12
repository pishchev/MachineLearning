import metrics
import utils
import matplotlib.pyplot as plt

def ROCSTEP(probe_predicts):
    points = []
    curx = 0
    cury = 0
    area = 0

    neg_step = 1/utils.calcVals(probe_predicts, 1)
    pos_step = 1/utils.calcVals(probe_predicts, 0)

    points.append([curx, cury])
    for i in probe_predicts:
        if (i[1] == 0):
            cury += pos_step
        else:
            curx += neg_step
            area += cury * neg_step
        points.append([curx, cury])

    print('Area ROCSTEP = %0.2f' % area)
    plt.title("ROC (Step method)")
    plt.plot(*zip(*points))
    plt.show()

def ROC(probe_predicts):
    points = list(map(lambda x: [metrics.FPR(probe_predicts, x[0]), metrics.TPR(probe_predicts, x[0])], probe_predicts))
    points.insert(0, [0,0]) 
    plt.title("ROC")
    print('Area ROC = %0.2f' % utils.areaCalc(points))
    plt.plot(*zip(*points))
    plt.show()
    
def PR(probe_predicts):
    points = list(map(lambda x: [metrics.recall(probe_predicts, x[0]), metrics.precision(probe_predicts, x[0])], probe_predicts))
    points.insert(0,[0,1])
    plt.title("PR")
    print('Area PR = %0.2f' % utils.areaCalc(points))
    plt.plot(*zip(*points))
    plt.show()