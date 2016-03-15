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

def PrelimMatrix(filename, N, f, x, y, unFlatRA, unFlatDEC):   #Stores all x, y, unflattened RA and DEC of reference stars from txt into a np.array
    for i in range(N):
        x_coor=float(f.readline())
        y_coor=float(f.readline())
        x[i]=x_coor
        y[i]=y_coor
        unflatRA=decimalRA(float(f.readline()), float(f.readline()), float(f.readline()))
        unflatDEC=decimalDEC(float(f.readline()), float(f.readline()), float(f.readline()))
        unFlatRA[i]=unflatRA
        unFlatDEC[i]=unflatDEC

def NextMatrix(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D, H): #Flattens the RA and DEC of stars and stores it into a np.array
    for i in range(N):
        H[i] = sin(unFlatDEC[i])*sin(D) + cos(unFlatDEC[i])*cos(D)*cos(unFlatRA[i] - A)
        FlatRA[i] = (cos(unFlatDEC[i])*sin(unFlatRA[i] - A))/H[i]
        FlatDEC[i] = (sin(unFlatDEC[i])*cos(D) - cos(unFlatDEC[i])*sin(D)*cos(unFlatRA[i] - A))/H[i]


def MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D):  #constructs the column matrix and the 3x3 matrix and returns the plate constants.
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

    return np.array([plateRA, plateDEC])

def unflatten(RA, DEC, filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D):  #This functions takes in flattened RA and spits out the unflattened RA and DEC
    Alpha = cos(D)-DEC*sin(D)
    newRA = atan(RA/Alpha) + A
    Beta = sqrt(RA**2 + Alpha**2)
    newDEC = (sin(D) + DEC*cos(D))/Beta
    return [newRA, newDEC]

def uncertainty(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D):
    b1 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0][0])
    a11 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0][1])
    a12 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0][2])
    b2 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1][0])
    a21 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1][1])
    a22 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1][2])

    uncertaintyRA = np.array(range(N), dtype=float)
    uncertaintyDEC = np.array(range(N), dtype=float)
    deltas = np.array(range(N), dtype=float)
    alphas = np.array(range(N), dtype=float)
    deltaun = np.array(range(N), dtype=float)
    alphaun =np.array(range(N), dtype=float)
    for i in range(N):
        alphas[i]=(b1) + (a11)*x[i] + (a12)*y[i]
        deltas[i]=(b2) + (a21)*x[i] + (a22)*y[i]
        alphaun[i] = unflatten(alphas[i], deltas[i], filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0]
        deltaun[i] = unflatten(alphas[i], deltas[i], filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1]
        uncertaintyRA[i] = (unFlatRA[i] - alphaun[i])**2
        uncertaintyDEC[i] = (unFlatDEC[i] - deltaun[i])**2
    RAvalue=((uncertaintyRA.sum()/(N-3))**0.5)/pi*180*3600
    DECvalue=((uncertaintyDEC.sum()/(N-3))**0.5)/pi*180*3600
    return RAvalue, DECvalue

def asteroid(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D):
    b1 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0][0])
    a11 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0][1])
    a12 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[0][2])
    b2 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1][0])
    a21 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1][1])
    a22 = float(MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)[1][2])

    x_ast=722.996971  #The X and Y coordinate of the asteroid is entered here.
    y_ast=352.997467
    RA_asteroid=(b1)+(a11)*(x_ast)+(a12)*(y_ast)
    DEC_asteroid=(b2)+(a21)*(x_ast)+(a22)*(y_ast)
    return unflatten(RA_asteroid, DEC_asteroid, filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)

    
def LSPR(filename):
    f=open(filename, "r")
    N = int(f.readline())
    FlatRA=np.array(range(N), dtype=float)
    FlatDEC=np.array(range(N), dtype=float)
    unFlatRA=np.array(range(N), dtype=float)
    unFlatDEC=np.array(range(N), dtype=float)
    x=np.array(range(N), dtype=float)
    y=np.array(range(N), dtype=float)
    H=np.array(range(N), dtype=float)
    
    PrelimMatrix(filename, N, f, x, y, unFlatRA, unFlatDEC)

    A = unFlatRA.sum()/float(N)
    D = unFlatDEC.sum()/float(N)

    NextMatrix(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D, H)
    print "Plate Constants:"
    print MatrixSetup(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)
    print "Uncertainty in arcseconds (RA, DEC):"
    print uncertainty(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)
    print "Calculated Asteroid coordinates:"
    print asteroid(filename, N, x, y, unFlatRA, unFlatDEC, FlatRA, FlatDEC, A, D)
LSPR("real.txt")


