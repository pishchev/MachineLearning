def thx_python_for_sort(array):
    for x in range(len(array)):
        for y in range(0, len(array) - x - 1):
            if array[y][0] < array[y+1][0]:
                s = array[y].copy()
                array[y] = array[y+1].copy()
                array[y+1] = s.copy()
                
def calcVals(arr, val):
    sum = 0
    for i in arr:
        sum += 1 if i[1] == val else 0
    return sum

def areaCalc(points):
    sum = 0
    for i in range(len(points) -1):
        sum += ((points[i][1] + points[i+1][1]) / 2) * (points[i+1][0] - points[i][0])
    return sum