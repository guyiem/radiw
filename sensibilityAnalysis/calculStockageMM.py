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


pmin = npy.array([ 1620 , 1650 , 0.5 , 0.0001 , 5 , 50 ] )
pmax = npy.array([ 1660 , 1750 , 1.5 , 0.005 , 50 , 150 ] )

ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(pmax[0]-pmin[0]) + pmin[0]
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
echants[:,4] = echants[:,4]*(50-5) + 5
echants[:,5] = echants[:,5]*(150-50) + 50


nm = "5"
rc2000 = pickle.load(open("donnees/rc1000e_m"+nm+"_2kHz.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000e_m"+nm+"_5kHz.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000e_m"+nm+"_7kHz.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000e_m"+nm+"_9kHz.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000e_m"+nm+"_11kHz.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000e_m"+nm+"_13kHz.pick","rb"))

Freqs = [2000 , 5000 , 7000 , 9000 , 11000 , 13000]
RCB = [ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ]

RCB = npy.array(RCB)

CPM = []
for kf,freq in enumerate(Freqs):
    print(kf)
    CPM.append( ChaosPolynomialsModel3(pmin,pmax,echants,RCB[kf]) )
pickle.dump(CPM,open("ensModeles"+nm+".pick","wb"))
