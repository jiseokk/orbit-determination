from __future__ import division
import numpy as np
from math import *

def findQuadrant(sine, cosine):
    theta = asin(sine)
    if cosine < 0:
        theta = pi - theta
    return theta

r = np.array([0.244, 2.17, -0.445])
rDot = np.array([-0.731, -0.0041, 0.0502])
t = 2451545 - 2456842.5

def orbitalElements(r, rDot, t):
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
    M = (M_0 + n*t) % (2*pi)
    
    return a, e, I/pi*180, Omega/pi*180, (w/pi*180) % (360), M/pi*180

print orbitalElements(r, rDot, t)


