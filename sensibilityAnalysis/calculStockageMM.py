#coding: utf-8

import numpy as npy
import numpy.matlib
import pickle
import pandas as pds
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *

# TODO : PASS AUTOMICALLY THE BOUNDS PARAMETERS

# script wich computes, once we have stored a set of correlation radius for a latin hyper cube sampling, the chaos polynomials coefficient of the meta-model, and store the chaos polynomial model


# choice of frequencies we're going to compute metamodels
Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]

# parameters min and max for the metamodels
# WARNING : FOR THE MOMENT, WE HAVE TO ENTER BY HAND HERE THE PARAMETERS WE USED FOR COMPUTING THE CORRELATION RADIUS ON THE LHCS. RISKS OF MISTAKE HERE !!!
pmin = npy.array([ 1530 , 1650 , 0.5 , 0.0001 , 5 , 50 ] )
pmax = npy.array([ 1630 , 1750 , 2.0 , 0.005 , 50 , 150 ] )

# loading of the latin hyper cube sampling (on 0-1)
ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(pmax[0]-pmin[0]) + pmin[0]
echants[:,1] = echants[:,1]*(pmax[1]-pmin[1]) + pmin[1]
echants[:,2] = echants[:,2]*(pmax[2]-pmin[2]) + pmin[2]
echants[:,3] = echants[:,3]*(pmax[3]-pmin[3]) + pmin[3]
echants[:,4] = echants[:,4]*(pmax[4]-pmin[4]) + pmin[4]
echants[:,5] = echants[:,5]*(pmax[5]-pmin[5]) + pmin[5]
# end of loading of the latin hyper cube sampling (on 0-1)

# loading of the computed radius of correlation for each frequencies
nm = "11"
rc2000 = pickle.load(open("donnees/rc1000e_m"+nm+"_2kHz.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000e_m"+nm+"_5kHz.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000e_m"+nm+"_7kHz.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000e_m"+nm+"_9kHz.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000e_m"+nm+"_11kHz.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000e_m"+nm+"_13kHz.pick","rb"))
RCB = [ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ]
RCB = npy.array(RCB)
# end of loading of the computed radius of correlation for each frequencies

# regression to compute the chaos polynomial coefficiants, and saving models
CPM = []
for kf,freq in enumerate(Freqs):
    print(kf)
    CPM.append( ChaosPolynomialsModel3(pmin,pmax,echants,RCB[kf]) )
pickle.dump(CPM,open("ensModeles"+nm+".pick","wb"))
