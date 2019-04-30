#coding: utf-8

import pickle
import sys
sys.path.append("../")
import numpy as npy
import pandas as pds
import Legendre

from ModeleVertical import *

ne = "1000"
echants_pds = pds.read_csv("lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(1640-1620) + 1620
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.59-0.59) + 0.59
echants[:,3] = echants[:,3]*(0.003-0.001) + 0.001
echants[:,4] = echants[:,4]*(50-10) + 10
echants[:,5] = echants[:,5]*(140-60) + 60

pcapteurs = 50 - 0.015*npy.arange(0,640)[::-1] # position of the hydrophones
dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones

Freqs = [2000, 5000, 7000, 9000, 11000, 13000]
for freq in Freqs:
    RC = []
    for kl,echant in enumerate(echants):
        print(kl)
        MV = ModeleVertical(108,1523,echant[0],1000,echant[1],echant[2],echant[3],echant[4],echant[5],freq)
        MSA = ModeleSourceAntenne(MV,50,200)
        RC.append(MSA.rayon_correlation_verticale(pcapteurs,dc))
    pickle.dump(RC,open("rc"+str(ne)+"_"+str(freq)+".pick","wb"))
