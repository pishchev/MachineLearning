import math

def Epanechnikov(r):
    return 3.0/4.0 * (1.0 - r*r)

def Kvartic(r):
    return 15.0/16.0 * (1 - r**2)**2

def Triangle(r):
    return 1.0 - abs(r)

def Gauss(r):
    return (2.0 * math.pi)**(-1.0/2.0) * math.exp(-1.0/2.0 * r**2)

def Rect(r):
    return 1.0/2.0