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
# vs = 1611
# rhos = 1700
# alpha = 1.2
# sigma = 0.0031
# lv = 27
# lh = 100
# vs = 1600
# rhos = 1700
# alpha = 1.5
# sigma = 0.003
# lv = 12
# lh = 100

po, rco = pickle.load(open("./donnees/rco_sac2016_1.pick","rb"))
# debug po-3
#params_optims[0] = 1595
po[2] = 1.15
po[4] = 35
po[3] = 0.0036
# fin debug
vs = po[0]
sigma = po[3]
lch = 100
lcv = po[4]
epsa = po[2]
rhos = 1700


params = [vs,rhos,alpha,sigma,lv,lh]

pcapteurs = 50 - 0.15*npy.arange(0,64)[::-1] # position of the hydrophones
dc =  0.15*npy.arange(0,64) # all the distance possible between the hydrophones
rct = []
for freq in Freqs:
    MV = ModeleVertical(108,1523,vs,1000,rhos,alpha,sigma,lv,lh,freq)
    MSA = ModeleSourceAntenne(MV,50,9000)
    rct.append(MSA.rayon_correlation_verticale(pcapteurs,dc))

print(npy.linalg.norm(rct-rce))
pickle.dump([params,rct],open("rco_sac2016_3.pick","wb"))
