from __future__ import division
import numpy as np
from math import *
import pyfits
import matplotlib.pyplot as plt
from time import sleep

def storeArray(N, xC, yC, FWHM, mag, f):
    for i in range(N):
        xC[i]=float(f.readline())
        yC[i]=float(f.readline())
        FWHM[i]=float(f.readline())
        mag[i]=float(f.readline())

def pixelExtract(N, xC, yC, FWHM, data):
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

def annulus(N, xC, yC, FWHM, data):
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

def subtract(N, star, backgroundStar):
    backgroundAverage=np.array(range(N), dtype=float)
    subtractedCount=np.array(range(N), dtype=float)
    for n in range(N):
        backgroundAverage[n]=backgroundStar[0][n] / backgroundStar[1][n]
        subtractedCount[n]=star[0][n] - star[1][n]*backgroundAverage[n]
    return subtractedCount

def linear(N, mag, subtractedCount):
    count = N
    logs = np.array(range(N), dtype=float)
    for i in range(N):
        logs[i] = np.log10(subtractedCount[i])
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
    
def asteroid(data):
    pixelCount = 0
    numberOfPixels = 0
    xtotal = np.array(range(data.shape[1]))
    ytotal = np.array(range(data.shape[0]))
    asteroid_x = 740.06
    asteroid_y = 369.24
    asteroid_FWHM = 4.48
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
    f=open(filename, "r")
    data=pyfits.getdata(FITS)
    N = int(f.readline())
    xC=np.array(range(N), dtype=float)
    yC=np.array(range(N), dtype=float)
    FWHM=np.array(range(N), dtype=float)
    mag=np.array(range(N), dtype=float)

    storeArray(N, xC, yC, FWHM, mag, f)

    star = pixelExtract(N, xC, yC, FWHM, data)

    backgroundStar = annulus(N, xC, yC, FWHM, data)

    subtractedCount = subtract(N, star, backgroundStar)
    
    regression = linear(N, mag, subtractedCount)

    print "Slope is", regression[0]
    print "Y-intercept is", regression[1]
    
    asteroidMag = regression[0] * np.log10(asteroid(data)) + regression[1]
    print "Asteroid magnitude is", asteroidMag

    backgroundPerPixel = backgroundStar[0].sum() / backgroundStar[1].sum()
    constant = (1.28)**(-1)
    backgroundMag = regression[0] * np.log10(backgroundPerPixel*constant*constant) + regression[1]
    print "Background mag per arcsec^2 is", backgroundMag
    print "X-Y entries:"
    print regression[2]
    print regression[3]
    print subtractedCount
    
    x = np.arange(0.0, 6.0, 0.01)
    y = regression[0]*x + regression[1]
    plt.plot(regression[2], regression[3], 'ro')
    plt.plot(x, y, lw=2)
    plt.axis([0,6,10,20])
    plt.show()

which = input("1 for NOMAD and 2 for UCAC3")

if which == 1:
    filename = "NOMAD.txt"
else:
    filename = "UCAC3.txt"
    
FITS = "second.FIT"

photometry(filename, FITS)
