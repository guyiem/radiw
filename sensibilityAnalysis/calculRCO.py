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
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))
vs = 1630
rhos = 1700
alpha = 1.09
sigma = 0.002
lv = 30
lh = 100

pcapteurs = 50 - 0.15*npy.arange(0,64)[::-1] # position of the hydrophones
dc =  0.15*npy.arange(0,64) # all the distance possible between the hydrophones
rct = []
for freq in Freqs:
    MV = ModeleVertical(108,1523,vs,1000,rhos,alpha,sigma,lv,lh,freq)
    MSA = ModeleSourceAntenne(MV,50,9000)
    rct.append(MSA.rayon_correlation_verticale(pcapteurs,dc))
pickle.dump(rct,open("rco_sac2016_vs"+str(vs)+"_epsa"+.str(alpha)+"_sig"+str(sigma)+"_lv"+str(lv)+".pick","wb"))
