from __future__ import division
import numpy as np
from math import *

def decimalRA(hour, minute, second): #turns (h,m,s) into radians
    decimalHour = hour + minute/60 + second/3600
    radRA = decimalHour/24*2*pi
    return radRA


def decimalDEC(degree, arcmin, arcsec): #turns (deg, min, sec) into radians
    if degree < 0:
        arcmin = (-1) * arcmin
        arcsec = (-1) * arcsec
    decimalDegree = degree + arcmin/60 + arcsec/3600
    radDEC = decimalDegree/180*pi
    return radDEC

def determinant(M):
    M = np.array(M.copy(), dtype=float)
    det = 0
    i = 0
    for j in range(3):
        iset=range(3)
        jset=range(3)
        iset.remove(i)
        jset.remove(j)
        det = det + ((-1)**(i+j)) * M[i,j] * (M[iset[0],jset[0]]*M[iset[1],jset[1]] - M[iset[1],jset[0]]*M[iset[0],jset[1]])
    return det

def inverse(M):
    M = np.array(M.copy(), dtype=float)
    N = np.array(M.copy(), dtype=float)
    for i in range(3):
        for j in range(3):
            iset=range(3)
            jset=range(3)
            iset.remove(i)
            jset.remove(j)
            N[i,j]=((-1)**(i+j)) * (M[jset[0],iset[0]] * M[jset[1],iset[1]] - M[jset[1],iset[0]] * M[jset[0],iset[1]]) / determinant(M)
    return N

def LSPRmatrix(filename): #contains flattening yet!
    f = open(filename, "r")
    
    N = int(f.readline())


    FlatRA=np.array(range(N), dtype=float)
    FlatDEC=np.array(range(N), dtype=float)
    unFlatRA=np.array(range(N), dtype=float)
    unFlatDEC=np.array(range(N), dtype=float)
    x=np.array(range(N), dtype=float)
    y=np.array(range(N), dtype=float)
    
    for i in range(N):
        x_coor=float(f.readline())
        y_coor=float(f.readline())
        x[i]=x_coor
        y[i]=y_coor
        unflatRA=decimalRA(float(f.readline()), float(f.readline()), float(f.readline()))
        unflatDEC=decimalDEC(float(f.readline()), float(f.readline()), float(f.readline()))
        unFlatRA[i]=unflatRA
        unFlatDEC[i]=unflatDEC

    A = unFlatRA.sum()/float(N)
    D = unFlatDEC.sum()/float(N)

    for i in range(N):
        H = sin(unFlatDEC[i])*sin(D) + cos(unFlatDEC[i])*sin(D)*cos(unFlatRA[i] - A)
        FlatRA[i] = (cos(unFlatDEC[i])*sin(unFlatRA[i] - A))/H
        FlatDEC[i] = (sin(unFlatDEC[i])*cos(D) - cos(unFlatDEC[i])*sin(D)*cos(unFlatRA[i] - A))/H 
    
    RAtimesX=np.dot(FlatRA, x.transpose())
    RAtimesY=np.dot(FlatRA, y.transpose())
    DECtimesX=np.dot(FlatDEC, x.transpose())
    DECtimesY=np.dot(FlatDEC, y.transpose())
    xSum=x.sum()
    ySum=y.sum()
    xSquared=np.dot(x, x.transpose())
    ySquared=np.dot(y, y.transpose())
    xtimesy=np.dot(x, y.transpose())
    RAcolumnMatrix=np.array([[FlatRA.sum()],[RAtimesX],[RAtimesY]])
    DECcolumnMatrix=np.array([[FlatDEC.sum()],[DECtimesX],[DECtimesY]])
    fullMatrix=np.array([[float(N), xSum, ySum], [xSum, xSquared, xtimesy], [ySum, xtimesy, ySquared]])

    plateRA=np.dot(inverse(fullMatrix), RAcolumnMatrix)
    plateDEC=np.dot(inverse(fullMatrix), DECcolumnMatrix)

    return plateRA, plateDEC

print LSPRmatrix("starsXY.txt")

def unflatten(RA,DEC):
    Alpha = cos(D)-DEC*sin(D)
    newRA = atan(RA/Alpha) + A
    return newRA
    Beta = sqrt(RA**2 + Alpha**2)
    newDEC = (sin(D) + DEC*cos(D))/Beta
    return newDEC


