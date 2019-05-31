#coding: utf-8

import numpy as npy
import numpy.matlib
import pickle
import pandas as pds
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *

# variable parameters in sensibility models study
vs = 1599
alpha = 1.51
sigma = 0.00299
lv = 12.1
# end variable parameters in sensibility models study
name = "donnees/rco_sac2016_2.pick" # name to save correlations radius and paramss


Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))

rhos = 1700
lh = 100
params = [vs,rhos,alpha,sigma,lv,lh]

pcapteurs = 50 - 0.015*npy.arange(0,640)[::-1] # position of the hydrophones
dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones
rct = []
for freq in Freqs:
    print(freq)
    MV = ModeleVertical(108,1523,vs,1000,rhos,alpha,sigma,lv,lh,freq)
    MSA = ModeleSourceAntenne(MV,50,9000)
    rct.append(MSA.rayon_correlation_verticale(pcapteurs,dc))

print(npy.linalg.norm(rct-rce))
pickle.dump([params,rct],open(name,"wb"))
