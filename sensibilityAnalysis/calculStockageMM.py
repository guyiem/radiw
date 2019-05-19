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

ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(1650-1600) + 1600
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
echants[:,4] = echants[:,4]*(50-5) + 5
echants[:,5] = echants[:,5]*(150-50) + 50

rc2000 = pickle.load(open("donnees/rc1000e_m1_2kHz.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000e_m1_5kHz.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000e_m1_7kHz.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000e_m1_9kHz.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000e_m1_11kHz.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000e_m1_13kHz.pick","rb"))

Freqs = [2000 , 5000 , 7000 , 9000 , 11000 , 13000]
RCB = [ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ]

CPM = []
for kf,freq in enumerate(Freqs):
    CPM.append( ChaosPolynomialsModel3(pmin,pmax,echants,RCB[kf]) )
pickle.dump(CPM,open("toto.pick","wb"))
