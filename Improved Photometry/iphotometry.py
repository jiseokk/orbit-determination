from __future__ import division
import numpy as np
from math import *
import pyfits
import matplotlib.pyplot as plt
from time import sleep

def storeArray(filename, N, xC, yC, FWHM, mag, f):
    for i in range(N):
        xC[i]=float(f.readline())
        yC[i]=float(f.readline())
        FWHM[i]=float(f.readline())
        mag[i]=float(f.readline())

def pixelExtract(filename, N, xC, yC, FWHM, data):
    pixelCount = np.array(range(N), dtype=float)
    numberOfPixels = np.array(range(N), dtype=float)
    xtotal = np.array(range(data.shape[1]))
    ytotal = np.array(range(data.shape[0]))
    for n in range(N):
        for i in xtotal[int(xC[n])-int(1.3*FWHM[n]):int(xC[n])+int(1.3*FWHM[n])]:
            for j in ytotal[int(yC[n])-int(1.3*FWHM[n]):int(yC[n])+int(1.3*FWHM[n])]:
                if sqrt((i-xC[n])**2+(j-yC[n])**2) < (FWHM[n]):
                    pixelCount[n] = pixelCount[n] + data[j, i]
                    numberOfPixels[n] = numberOfPixels[n] + 1
    return [pixelCount, numberOfPixels]

def annulus(filename, N, xC, yC, FWHM, data):
    pixelCount = np.array(range(N), dtype=float)
    numberOfPixels = np.array(range(N), dtype=float)
    xtotal = np.array(range(data.shape[1]))
    ytotal = np.array(range(data.shape[0]))
    for n in range(N):
        for i in xtotal[int(xC[n])-int(3*FWHM[n]):int(xC[n])+int(3*FWHM[n])]:
            for j in ytotal[int(yC[n])-int(3*FWHM[n]):int(yC[n])+int(3*FWHM[n])]:
                if sqrt((i-xC[n])**2+(j-yC[n])**2) > (FWHM[n]*1.5) and sqrt((i-xC[n])**2+(j-yC[n])**2) < (FWHM[n]*2.5):
                    pixelCount[n] = pixelCount[n] + data[j, i]
                    numberOfPixels[n] = numberOfPixels[n] + 1
    return [pixelCount, numberOfPixels]

def subtract(filename, N, xC, yC, FWHM, data):
    backgroundAverage=np.array(range(N), dtype=float)
    subtractedCount=np.array(range(N), dtype=float)
    for n in range(N):
        backgroundAverage[n]=annulus(filename, N, xC, yC, FWHM, data)[0][n] / annulus(filename, N, xC, yC, FWHM, data)[1][n]
        subtractedCount[n]=pixelExtract(filename, N, xC, yC, FWHM, data)[0][n] - pixelExtract(filename, N, xC, yC, FWHM, data)[1][n]*backgroundAverage[n]
    return subtractedCount

def linear(filename, N, xC, yC, FWHM, data, mag):
    count = N
    logs = np.array(range(N), dtype=float)
    for i in range(N):
        logs[i] = np.log10(subtract(filename, N, xC, yC, FWHM, data)[i])
    SumX = logs.sum()
    SumY = mag.sum()
    logsq = np.array(range(N), dtype=float)
    for i in range(N):
        logsq[i] = logs[i]**2
    SumX2 = logsq.sum()
    logmag = np.array(range(N), dtype=float)
    for i in range(N):
        logmag[i] = logs[i]*mag[i]
    SumXY = logmag.sum()

    XMean = SumX / count
    YMean = SumY / count
    Slope = (SumXY - SumX * YMean) / (SumX2 - SumX*XMean)
    YInt = YMean - Slope*XMean
    
    return [Slope, YInt, logs, mag]
    

def asteroid(filename, N, xC, yC, FWHM, data, mag):
    pixelCount = 0
    numberOfPixels = 0
    xtotal = np.array(range(data.shape[1]))
    ytotal = np.array(range(data.shape[0]))
    asteroid_x = 724.11
    asteroid_y = 354.00
    asteroid_FWHM = 5.44
    for i in xtotal[int(asteroid_x)-3*int(asteroid_FWHM):int(asteroid_x)+3*int(asteroid_FWHM)]:
        for j in ytotal[int(asteroid_y)-3*int(asteroid_FWHM):int(asteroid_y)+3*int(asteroid_FWHM)]:
            if sqrt((i-asteroid_x)**2+(j-asteroid_y)**2) < (asteroid_FWHM):
                pixelCount = pixelCount + data[j, i]
                numberOfPixels = numberOfPixels + 1


    backgroundCount = 0
    backgroundPixels = 0
    for i in xtotal[int(asteroid_x)-5*int(asteroid_FWHM):int(asteroid_x)+5*int(asteroid_FWHM)]:
        for j in ytotal[int(asteroid_y)-5*int(asteroid_FWHM):int(asteroid_y)+5*int(asteroid_FWHM)]:
            if sqrt((i-asteroid_x)**2+(j-asteroid_y)**2) > (asteroid_FWHM/2*2.5) and sqrt((i-asteroid_x)**2+(j-asteroid_y)**2) < (asteroid_FWHM/2*3.5):
                backgroundCount = backgroundCount + data[j, i]
                backgroundPixels = backgroundPixels + 1
 

    asteroidPixel = pixelCount -  numberOfPixels * backgroundCount / backgroundPixels

    return asteroidPixel



def photometry(filename, FITS):
    print "This program takes some time to run. Your patience would be greatly appreciated."
    sleep(3)
    print "This program,"
    sleep(1)
    print "however,"
    sleep(2)
    print "won't let you down."
    sleep(2)
    print "Cuz it's super ¡rock-star!"
    f=open(filename, "r")
    data=pyfits.getdata(FITS)
    N = int(f.readline())
    xC=np.array(range(N), dtype=float)
    yC=np.array(range(N), dtype=float)
    FWHM=np.array(range(N), dtype=float)
    mag=np.array(range(N), dtype=float)
    
    storeArray(filename, N, xC, yC, FWHM, mag, f)

    regression = linear(filename, N, xC, yC, FWHM, data, mag)

    x = np.arange(0.0, 6.0, 0.01)
    y = regression[0]*x + regression[1]
    plt.plot(regression[2], regression[3], 'ro')
    plt.plot(x, y, lw=2)
    plt.axis([0,6,10,20])
    plt.show()

    asteroidMag = regression[0] * np.log10(asteroid(filename, N, xC, yC, FWHM, data, mag)) + regression[1]
    print asteroidMag

    skyBackground = annulus(filename, N, xC, yC, FWHM, data)
    backgroundPerPixel = skyBackground[0].sum() / skyBackground[1].sum()
    constant = (1.28)**(-1)
    backgroundMag = regression[0] * np.log10(backgroundPerPixel*constant*constant) + regression[1]
    print backgroundMag
    print regression
    
filename = "c2fit.txt"
FITS = "C2.FIT"

photometry(filename, FITS)
