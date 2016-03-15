#The Orbit Determination Code
#Ji Seok Kim
#Team 10: Millenium Aquila
#Asteroid: 2005 TR15

from __future__ import division
import numpy as np
from math import *
import random as rd
#from visual import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def decimalRA(hour, minute, second): #turns RA in (hour,min,sec) into radians
    decimalHour = hour + minute/60 + second/3600
    radRA = decimalHour/24*2*pi
    return radRA

def decimalDEC(degree, arcmin, arcsec): #turns DEC in (deg,arcm,arcs) into radians
    if degree < 0:
        arcmin = (-1) * arcmin
        arcsec = (-1) * arcsec
    decimalDegree = degree + arcmin/60 + arcsec/3600
    radDEC = decimalDegree/180*pi
    return radDEC

def julian(y, m, d): #calculates julian date given (year, month, decimal day)
    day = int(d)
    hour = d - day
    first = 367 * y
    second = int(7 * (y+int((m+9)/12))/4)
    third = int(275 * m / 9)
    fourth = day
    fifth = 1721013.5
    J_0 = first - second + third + fourth + fifth
    JD = J_0 + hour
    return JD

#You might be wondering what "Combo" is.
#Since we have 5 nights of data,
#I use 'Combo' as a list that specifies which three data sets we use.
#For example, using 'Combo = [0,2,4]' will determine the orbit using first, third, and fifth observations.

def ro_hat(RA, DEC, Combo): #takes RA's DEC's gives ro_hat vectors
    roHat = np.zeros(shape=(3,3))
    for i in range(3):
        roHat[i][0] = cos(DEC[Combo[i]]) * cos(RA[Combo[i]])
        roHat[i][1] = cos(DEC[Combo[i]]) * sin(RA[Combo[i]])
        roHat[i][2] = sin(DEC[Combo[i]])
    return roHat

def tao(time, Combo): #takes the julian dates and gives tao (time intervals in units of gaussian day)
    t = [0,0,0]
    for i in range(3):
        t[i] = time[Combo[i]]
        
    mu = 0.01720209895
    tao0=mu * (t[2] - t[0])
    tao1=mu * (t[0] - t[1])
    tao3=mu * (t[2] - t[1])
    return [tao1, tao3, tao0]

def aConstant(taos, Combo): #initial a1, a3 constants
    tao1 = taos[0]
    tao3 = taos[1]
    tao0 = taos[2]
    a1 = tao3 / tao0
    a3 = (-1) * tao1 / tao0
    return [a1, a3]

def ro_mag(roHat, a, sVector, Combo):#Calculates the magnitude of ro vectors using Cramer's Rule
    a1 = a[0]
    a3 = a[1]
    B = sVector[Combo[1]] - a1 * sVector[Combo[0]] - a3 * sVector[Combo[2]]
    A1 = (-1) * a1 * roHat[0]
    A2 = roHat[1]
    A3 = (-1) * a3 * roHat[2]

    ro1 = np.dot( B , np.cross( A2 , A3)) / np.dot( A1 , np.cross( A2 , A3))
    ro2 = np.dot( B , np.cross( A1 , A3)) / np.dot( A2 , np.cross( A1 , A3))
    ro3 = np.dot( B , np.cross( A1 , A2)) / np.dot( A3 , np.cross( A1 , A2))

    #for checking the B vector, you can do this:
    #C = ro1 * A1 + ro2 * A2 + ro3 * A3  
    #return B, C

    return [ro1, ro2, ro3]

def rVector(roMag, roHat, sVector, a, taos, Combo): #Gives initial position vectors and triangular-approximated velocity vector
    ro = np.zeros(shape=(3,3))
    for i in range(3):
        for j in range(3):
            ro[i][j] = roMag[i] * roHat[i][j]
    r1 = ro[0] - sVector[Combo[0]]
    r3 = ro[2] - sVector[Combo[2]]
    r2 = ro[1] - sVector[Combo[1]]

    a1 = a[0]
    a3 = a[1]

    #for checking r2 value:
    #r2_check = a1 * r1 + a3 * r3

    tao1 = taos[0]
    tao3 = taos[1]
    tao0 = taos[2]
    
    r2dot = 0.5 * ( (-1) * (r2 - r1) / tao1 + (r3 - r2) / tao3 )
    return [r2, r2dot, r1, r3]

def f(rVec, tao): #The 'f' function that takes r2, r2Dot, tao as input and gives out the taylor series
    r2 = rVec[0]
    r2Dot = rVec[1]
    r2Mag = np.linalg.norm(r2)
    r2DotMag = np.linalg.norm(r2Dot)
    first = 1
    second = tao**2 / (2 * r2Mag**3)
    third = tao**3 * (np.dot( r2 , r2Dot )) / (2 * r2Mag**5)
    fourth = ( 3 * (np.dot( r2, r2Dot ) / r2Mag**2 - 1 / r2Mag**3) / r2Mag**3)
    fifth = 15 * (np.dot( r2, r2Dot ))**2 / r2Mag**2
    sixth = 1 / r2Mag**6
    return 1 - second + third #+ tao**4 / 24 * (fourth - fifth + sixth)
    #the terms that are commented out are extra terms of the Taylor series, which surprisingly makes the process worse.
    #Perhaps there is a mistake in the expression.

def g(rVec, tao): #The 'g' function
    r2 = rVec[0]
    r2Dot = rVec[1]
    r2Mag = np.linalg.norm(r2)
    r2DotMag = np.linalg.norm(r2Dot)
    first = tao
    second = tao**3 / (6 * r2Mag**3)
    third = tao**4 * (np.dot( r2 , r2Dot )) / (4 * r2Mag**5) 
    return first - second #+ third


def newA(f0, g0, f1, g1):#Takes the f and g functions and gives new a1, a3 constants
    a1 = g1 / (f0 * g1 - f1 * g0)
    a3 = (-1) * g0 / (f0 * g1 - f1 * g0)
    return [a1, a3]

def equaToEclip(r): #Cartesian coordinate rotation from equatorial to ecliptic
    x = r[0]
    y = r[1]
    z = r[2]
    
    E = (23.43333333333333333333333333) / 180 * pi

    x_ = x
    y_ = y*cos(E) + z*sin(E)
    z_ = z*cos(E) - y*sin(E)

    return np.array([x_, y_, z_])


def findQuadrant(sine, cosine): #Takes sine, cosine of an angle and gives the angle in the right quadrant
    theta = asin(sine)
    if cosine < 0:
        theta = pi - theta
    return theta

def orbitalElements(r, rDot, t, epoch): #takes position, velocity vectors and gives the six orbital elements (Mean Anomaly at time = t)
    h = np.cross(r, rDot)
    h_mag = np.linalg.norm(h)
    
    a = (2/np.linalg.norm(r)-np.dot(rDot, rDot))**(-1)
    
    e = (1-((h_mag)**2)/a)**0.5
    
    I = acos(h[2]/(h_mag))
    
    sin_omega = h[0] / h_mag / sin(I)
    cos_omega = (-1)*h[1]/h_mag / sin(I)
    Omega = findQuadrant(sin_omega, cos_omega)
    

    cos_wf = (r[0]*cos(Omega)+r[1]*sin(Omega))/np.linalg.norm(r)
    sin_wf = r[2] / sin(I) / np.linalg.norm(r)
    wf = findQuadrant(sin_wf, cos_wf)

    cos_f = (a*(1-e**2)/np.linalg.norm(r)-1)/e
    sin_f = a*(1-e**2) / h_mag * np.dot(r, rDot) / e / np.linalg.norm(r)
    f = findQuadrant(sin_f, cos_f)
    
    w = wf - f
       
    k = 0.01720209895
    n = k/(a**(3/2))

    E_0 = acos((1-np.linalg.norm(r)/a)/e)
    if f<0 or f > pi:
        E_0 = -E_0
        
    M_0 = E_0 - e*sin(E_0)
    M = (M_0 + n*(epoch - t)) % (2*pi)
    
    return [a, e, I/pi*180, Omega/pi*180, (w/pi*180) % (360), M/pi*180]





def dataSetup(): #Takes the .txt input file and stores data input, as a float, as distincts elements of a list
    f = open("CTInput.txt", "r")
    lines = []
    line = f.readline
    
    while line: #each line, as a string, becomes an element in the 'lines' list
        line = f.readline()
        line = line.replace(':', ' ') 
        lines.append(line)

    del lines[-1] #There is an extra unneeded element in 'lines' list, so I delete it here.

    N = len(lines) #For each line, the data input gets separated by spaces.
    for i in range(N):
        lines[i] = lines[i].split()

    M = len(lines[0])
    for i in range(N):
        for j in range(M):
            lines[i][j] = float(lines[i][j]) #Now, make the strings into floats.
            
    return lines

def OD(Combo): #The actual OD which takes integrates all the functions above.
    lines = dataSetup()
    N = len(lines)

    #Now, we set up numpy arrays corresponding to each type of data input: RA, DEC, time, Earth-Sun vector.
    pRA = np.zeros(shape=(N,3))
    pDEC = np.zeros(shape=(N,3))
    ptime = np.zeros(shape=(N,6))
    sVector = np.zeros(shape=(N,3))

    decimalTime = np.zeros(shape=(N,3))

    time = np.array(range(N), dtype=float)
    
    for i in range(N):
        for j in range(3):
            pRA[i][j] = lines[i][j]
            pDEC[i][j] = lines[i][j+3]
            sVector[i][j] = lines[i][j+12]
    for i in range(N):
        for j in range(6):
            ptime[i][j] = lines[i][j+6]
    for i in range(N):
        decimalTime[i][0] = ptime[i][0]
        decimalTime[i][1]= ptime[i][1]
        decimalTime[i][2] = ptime[i][2] + (ptime[i][3] + ptime[i][4] / 60 + ptime[i][5] / 3600) / 24 
    
    for i in range(N):
        time[i] = julian(decimalTime[i][0], decimalTime[i][1], decimalTime[i][2])

    RA = np.array(range(N), dtype=float)
    DEC = np.array(range(N), dtype=float)
    
    for i in range(N):
        RA[i] = decimalRA(pRA[i][0], pRA[i][1], pRA[i][2])
        DEC[i] = decimalDEC(pDEC[i][0], pDEC[i][1], pDEC[i][2])
    
    #We now use the data arrays to perform the mathematics.
    roHat = ro_hat(RA, DEC, Combo)
    taos = tao(time, Combo)
    
    a = aConstant(taos, Combo) 
    
    roMag = ro_mag(roHat, a, sVector, Combo)
    
    rVec = rVector(roMag, roHat, sVector, a, taos, Combo)
    
    #This is the first iteration.
    oldAConstant = a
    f0 = f(rVec, taos[0])
    g0 = g(rVec, taos[0])
    f1 = f(rVec, taos[1])
    g1 = g(rVec, taos[1])
    newAConstant = newA(f0, g0, f1, g1)
    roMag = ro_mag(roHat, newAConstant, sVector, Combo)
    rVec = rVector(roMag, roHat, sVector, a, taos, Combo)
    r2 = rVec[0]
    r1 = rVec[2]
    r3 = rVec[3]
        
    r2Dot_1 = (r3 - f1 * r2) / g1
    r2Dot_2 = (r1 - f0 * r2) / g0
    r2Dot = (r2Dot_1 + r2Dot_2) / 2
    rVec[1] = r2Dot

    #next iterations contained in the while loop
    count = 1
    while abs(oldAConstant[0] - newAConstant[0]) > 0 or abs(oldAConstant[1] - newAConstant[1]) > 0:
    #We run the while loop until python can't distinguish between the two successive constants. 
        oldAConstant = newAConstant
        #print oldAConstant
        f0 = f(rVec, taos[0])
        g0 = g(rVec, taos[0])
        f1 = f(rVec, taos[1])
        g1 = g(rVec, taos[1])
        newAConstant = newA(f0, g0, f1, g1)
        #print newAConstant
        
        roMag = ro_mag(roHat, newAConstant, sVector, Combo)
        rVec = rVector(roMag, roHat, sVector, a, taos, Combo)
        r2 = rVec[0]
        r1 = rVec[2]
        r3 = rVec[3]
        
        r2Dot_1 = (r3 - f1 * r2) / g1
        r2Dot_2 = (r1 - f0 * r2) / g0
        r2Dot = (r2Dot_1 + r2Dot_2) / 2
        rVec[1] = r2Dot
        count = count +1

    #print "That took", count, "iterations."
        
    #print "The modified 'a' constants are:", newAConstant
    #print r2, r2Dot

    r2_Eclip = equaToEclip(r2)
    r2Dot_Eclip = equaToEclip(r2Dot)

    #print "The position and velocity vectors in cartesian ecliptic coordaintes:"
    #print r2_Eclip, r2Dot_Eclip
    
    t=2456837.839203520
    epoch = 2456800.5
    return orbitalElements(r2_Eclip, r2Dot_Eclip, t, epoch)




def newOD(Combo, RA, DEC):
#This is a variant of the OD function that manually takes the RA and DEC as inputs.
#This fuction is needed later on when the RA and DEC of middle observation will be tweaked for uncertainty calculation.
    lines = dataSetup()
    N = len(lines)
    ptime = np.zeros(shape=(N,6))
    sVector = np.zeros(shape=(N,3))
    decimalTime = np.zeros(shape=(N,3))
    time = np.array(range(N), dtype=float)  
    for i in range(N):
        for j in range(3):
            sVector[i][j] = lines[i][j+12]
    for i in range(N):
        for j in range(6):
            ptime[i][j] = lines[i][j+6]
    for i in range(N):
        decimalTime[i][0] = ptime[i][0]
        decimalTime[i][1]= ptime[i][1]
        decimalTime[i][2] = ptime[i][2] + (ptime[i][3] + ptime[i][4] / 60 + ptime[i][5] / 3600) / 24  
    for i in range(N):
        time[i] = julian(decimalTime[i][0], decimalTime[i][1], decimalTime[i][2])
    roHat = ro_hat(RA, DEC, Combo)
    taos = tao(time, Combo)
    a = aConstant(taos, Combo) 
    roMag = ro_mag(roHat, a, sVector, Combo)
    rVec = rVector(roMag, roHat, sVector, a, taos, Combo)
    oldAConstant = a
    f0 = f(rVec, taos[0])
    g0 = g(rVec, taos[0])
    f1 = f(rVec, taos[1])
    g1 = g(rVec, taos[1])
    newAConstant = newA(f0, g0, f1, g1)
    roMag = ro_mag(roHat, newAConstant, sVector, Combo)
    rVec = rVector(roMag, roHat, sVector, a, taos, Combo)
    r2 = rVec[0]
    r1 = rVec[2]
    r3 = rVec[3] 
    r2Dot_1 = (r3 - f1 * r2) / g1
    r2Dot_2 = (r1 - f0 * r2) / g0
    r2Dot = (r2Dot_1 + r2Dot_2) / 2
    rVec[1] = r2Dot
    count = 1
    while abs(oldAConstant[0] - newAConstant[0]) > 0.00000000001 or abs(oldAConstant[1] - newAConstant[1]) > 0.00000000001:
        oldAConstant = newAConstant  
        f0 = f(rVec, taos[0])
        g0 = g(rVec, taos[0])
        f1 = f(rVec, taos[1])
        g1 = g(rVec, taos[1])
        newAConstant = newA(f0, g0, f1, g1)  
        roMag = ro_mag(roHat, newAConstant, sVector, Combo)
        rVec = rVector(roMag, roHat, sVector, a, taos, Combo)
        r2 = rVec[0]
        r1 = rVec[2]
        r3 = rVec[3]
        r2Dot_1 = (r3 - f1 * r2) / g1
        r2Dot_2 = (r1 - f0 * r2) / g0
        r2Dot = (r2Dot_1 + r2Dot_2) / 2
        rVec[1] = r2Dot
        count = count +1
    r2_Eclip = equaToEclip(r2)
    r2Dot_Eclip = equaToEclip(r2Dot)
    t=2456837.839203520
    epoch = 2456800.5 + 67.185093/3600/24
    return orbitalElements(r2_Eclip, r2Dot_Eclip, t, epoch)

print "The orbital elements are:"
print OD([0,1,2])
print "Please be patient while I calculate the uncertainty."


def uncertainty(n, Combo): #calculate uncertainty
    RA_error = 0.5214789432567722 #RA and DEC uncertainty from LSPR code
    DEC_error = 0.2774092111401255
    

    RA = [4.21669293,  4.25379499,  4.31646395,  4.48544891,  4.52537259] 
    DEC = [-0.06531264, -0.05680029, -0.04934288, -0.05615985, -0.06145595]

    theElements = []

    x = []
    y = []
    for i in range(n): #This loop will vary the RA and DEC of the middle observation by amount within the error. 
        RA[Combo[1]] = rd.normalvariate(RA[Combo[1]], RA_error / 3600 / 180 * pi)
        DEC[Combo[1]] = rd.normalvariate(DEC[Combo[1]], DEC_error / 3600 / 180 * pi)
        
        x.append(RA[Combo[1]])
        y.append(DEC[Combo[1]])
    
        theElements.append(newOD(Combo, RA, DEC))

    a = []
    e = []
    I = []
    Omega = []
    w = []                       
    M = []
    
    for i in range(n): #Make a list for each orbital element
        a.append(theElements[i][0])
        e.append(theElements[i][1])
        I.append(theElements[i][2])
        Omega.append(theElements[i][3])
        w.append(theElements[i][4])
        M.append(theElements[i][5])

    transposeElements = []
    transposeElements.append(a)
    transposeElements.append(e)
    transposeElements.append(I)
    transposeElements.append(Omega)
    transposeElements.append(w)
    transposeElements.append(M)

    #Sort the lists from smallest to greatest
    aSort = sorted(a) 
    eSort = sorted(e)
    ISort = sorted(I)
    OmegaSort = sorted(Omega)
    wSort = sorted(w)
    MSort = sorted(M)


    inquiry = input("If you would like to see exactly how the orbital elements vary according to variations in right ascension and declination, press 1. If not, then 2.")
    if inquiry == 1:   
        text = ["Distribution of a", "Distribution of e", "Distribution of I", "Distribution of Omega", "Distribution of w", "Distribution of M"]
        for i in range(6):
            fig = plt.figure()
            ax = Axes3D(fig)
        
            z = transposeElements[i]
            ax.plot(x, y, z, 'ro') #zdir='z', label='zs=0, zdir=z')
            plt.xlabel("RA of mid. observation")
            plt.ylabel("DEC of mid. observation")
            plt.suptitle(text[i])
            plt.show()
    else:
        0
    
    #Throw away the top 1% and bottom 1% of the data to account for extreme values of the normal distribution
    del aSort[0:int(n/100)]
    del aSort[int(n*99/100)-1:(n-1)]
    del eSort[0:int(n/100)]
    del eSort[int(n*99/100)-1:(n-1)]
    del ISort[0:int(n/100)]
    del ISort[int(n*99/100)-1:(n-1)]
    del OmegaSort[0:int(n/100)]
    del OmegaSort[int(n*99/100)-1:(n-1)]
    del wSort[0:int(n/100)]
    del wSort[int(n*99/100)-1:(n-1)]
    del MSort[0:int(n/100)]
    del MSort[int(n*99/100)-1:(n-1)]

    elements = OD([0,2,3])
    extrema_a = [abs(elements[0]-max(aSort)), abs(elements[0]-min(aSort))]
    extrema_e = [abs(elements[1]-max(eSort)), abs(elements[1]-min(eSort))]
    extrema_I = [abs(elements[2]-max(ISort)), abs(elements[2]-min(ISort))]
    extrema_Omega = [abs(elements[3]-max(OmegaSort)), abs(elements[3]-min(OmegaSort))]
    extrema_w = [abs(elements[4]-max(wSort)), abs(elements[4]-min(wSort))]
    extrema_M = [abs(elements[5]-max(MSort)), abs(elements[5]-min(MSort))]

    sigma_a = max(extrema_a)
    sigma_e = max(extrema_e)
    sigma_I = max(extrema_I)
    sigma_Omega = max(extrema_Omega)
    sigma_w = max(extrema_w)
    sigma_M = max(extrema_M)
    
    print sigma_a, sigma_e, sigma_I, sigma_Omega, sigma_w, sigma_M
        

uncertainty(1000, [0,1,2])

"""
def solvekep(M):
    Eguess = M
    Mguess = Eguess - e*sin(Eguess) 
    Mtrue = M
    while abs(Mguess - Mtrue) > 1e-004:
        Mguess = Eguess - e*sin(Eguess) 
        Eguess = Eguess - (Eguess - e*sin(Eguess) - Mtrue) / (1 - e*cos(Eguess))
    return Eguess

#def visualize(Combo):

elements = OD([0,2,3])
a = elements[0]
e = elements[1]
M = radians(elements[5]) 
Oprime = radians(elements[3]) 
iprime = radians(elements[2]) 
wprime = radians(elements[4])

sqrtmu = 0.01720209895
mu = sqrtmu**2
time = 0
period = sqrt(4*pi**2*a**3/mu) 
r1ecliptic = vector(0, 0, 0) 
Mtrue = 2*pi/period*(time) + M 
Etrue = solvekep(Mtrue)
position = np.array([a*cos(Etrue)-a*e, a*sqrt(1-e**2)*sin(Etrue), 0])
r_w = np.array([[cos(wprime), -sin(wprime), 0],[sin(wprime), cos(wprime), 0],[0, 0, 1]])
r_i = np.array([[1, 0, 0],[0, cos(iprime), -sin(iprime)],[0, sin(iprime), cos(iprime)]])
r_O = np.array([[cos(Oprime), -sin(Oprime), 0],[sin(Oprime), cos(Oprime), 0],[0, 0, 1]])

rotation = np.dot(r_w, r_i, r_O)
r1ecliptic= np.dot(rotation, position)

asteroid = sphere(pos=r1ecliptic*150, radius=(6), color=color.white) 
asteroid.trail = curve(color=color.white) 
sun = sphere(pos=(0,0,0), radius=(20), color=color.yellow)


a_ = 1.000278835080823E+00
e_ = 1.673450045254545E-02
M_ = radians(38.61544515) 
Oprime_ = radians(2.088838905406901E+02) 
iprime_ = radians(2.973582895408738E-03) 
wprime_ =  radians(2.556655171825315E+02)

sqrtmu = 0.01720209895
mu = sqrtmu**2
time_ = 0
period_ = sqrt(4*pi**2*a_**3/mu) 
r1ecliptic_ = vector(0, 0, 0) 
Mtrue_ = 2*pi/period*(time_) + M_ 
Etrue_ = solvekep(Mtrue_)
position_ = np.array([a_*cos(Etrue_)-a_*e_, a_*sqrt(1-e_**2)*sin(Etrue_), 0])
r_w_ = np.array([[cos(wprime_), -sin(wprime_), 0],[sin(wprime_), cos(wprime_), 0],[0, 0, 1]])
r_i_ = np.array([[1, 0, 0],[0, cos(iprime_), -sin(iprime_)],[0, sin(iprime_), cos(iprime_)]])
r_O_ = np.array([[cos(Oprime_), -sin(Oprime_), 0],[sin(Oprime_), cos(Oprime_), 0],[0, 0, 1]])


rotation_ = np.dot(r_w_, r_i_, r_O_)
r1ecliptic_= np.dot(rotation_, position_)

asteroid_ = sphere(pos=r1ecliptic_*150, radius=(10), color=color.blue) 
asteroid_.trail = curve(color=color.blue) 


     
while (1==1): 
    rate(300) 

    time = time + 0.1

    time_ = time_ + 0.1

    Mtrue = 2*pi/period*(time) + M 
    Etrue = solvekep(Mtrue)

    Mtrue_ = 2*pi/period_*(time_) + M_ 
    Etrue_ = solvekep(Mtrue_)

    position = np.array([a*cos(Etrue)-a*e, a*sqrt(1-e**2)*sin(Etrue), 0])

    position_ = np.array([a_*cos(Etrue_)-a_*e_, a_*sqrt(1-e_**2)*sin(Etrue_), 0])

    r_w = np.array([[cos(wprime), -sin(wprime), 0],[sin(wprime), cos(wprime), 0],[0, 0, 1]])
    r_i = np.array([[1, 0, 0],[0, cos(iprime), -sin(iprime)],[0, sin(iprime), cos(iprime)]])
    r_O = np.array([[cos(Oprime), -sin(Oprime), 0],[sin(Oprime), cos(Oprime), 0],[0, 0, 1]])

    r_w_ = np.array([[cos(wprime_), -sin(wprime_), 0],[sin(wprime_), cos(wprime_), 0],[0, 0, 1]])
    r_i_ = np.array([[1, 0, 0],[0, cos(iprime_), -sin(iprime_)],[0, sin(iprime_), cos(iprime_)]])
    r_O_ = np.array([[cos(Oprime_), -sin(Oprime_), 0],[sin(Oprime_), cos(Oprime_), 0],[0, 0, 1]])
    
    rotation = np.dot(r_w, r_i, r_O)
    r1ecliptic= np.dot(rotation, position)

    rotation_ = np.dot(r_w_, r_i_, r_O_)
    r1ecliptic_= np.dot(rotation_, position_)
    
    asteroid.pos = r1ecliptic*150
    asteroid.trail.append(pos=asteroid.pos)

    asteroid_.pos = r1ecliptic_*150
    asteroid_.trail.append(pos=asteroid_.pos)

"""



