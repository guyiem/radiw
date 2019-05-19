#coding: utf-8

import numpy as npy
import numpy.matlib
import pickle
import pandas as pds
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *

Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]

pcapteurs = 50 - 0.15*npy.arange(0,64)[::-1] # position of the hydrophones
dc =  0.15*npy.arange(0,64) # all the distance possible between the hydrophones

VS = npy.arange(1550,1650,10)
rhos = 1700
EPSA = npy.arange(0.1,3.0,0.1)
SIGMA = npy.arange(0.0005,0.02,0.0005)
LV = npy.arange(5,50,5)
lh = 100

nbe = len(VS) * len(EPSA) * len(SIGMA) * len(LV)
print(" nb échants : ",nbe)
print(" vs : ",len(VS))
print(" epsa : ",len(EPSA))
print(" sigma : ",len(SIGMA))
print(" lv : ",len(LV))

VV,EE,SS,LL = npy.meshgrid(VS,EPSA,SIGMA,LV)
echants = npy.vstack([ VV.ravel() , EE.ravel() , SS.ravel() , LL.ravel() ]).T

RCT = npy.zeros( (nbe , len(Freqs)) )
for ke,echant in enumerate(echants):
    print(ke, echant)
    for kf,freq in enumerate(Freqs):
        MV = ModeleVertical(108,1523,echant[0],1000,rhos,echant[1],echant[2],echant[3],lh,freq)
        MSA = ModeleSourceAntenne(MV,50,9000)
        RCT[ke,kf] = MSA.rayon_correlation_verticale(pcapteurs,dc)

pickle.dump([echants,RCT],open("optim.pick","wb"))
