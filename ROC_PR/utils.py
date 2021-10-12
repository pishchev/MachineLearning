from functools import reduce

def calcVals(arr, val):
    return reduce(lambda x, y: x+1 if y[1] == val else x, arr, 0)

def areaCalc(points):
    return reduce(lambda x,y: x + (y[0][1] + y[1][1])/2 * (y[1][0] - y[0][0]), zip(points[:-1], points[1:]), 0)
