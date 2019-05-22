#coding: utf-8

import pickle
import multiprocessing as mpr
from multiprocessing import Pool
import sys
sys.path.append("../")
import numpy as npy
import pandas as pds
import ChaosPolynomials

from ModeleVertical import *


ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
# echants[:,0] = echants[:,0]*(1650-1600) + 1600
# echants[:,1] = echants[:,1]*(1750-1650) + 1650
# echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
# echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
# echants[:,4] = echants[:,4]*(50-5) + 5
# echants[:,5] = echants[:,5]*(150-50) + 50

echants[:,0] = echants[:,0]*(1680-1630) + 1630
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
echants[:,4] = echants[:,4]*(50-5) + 5
echants[:,5] = echants[:,5]*(150-50) + 50

pcapteurs = 50 - 0.15*npy.arange(0,64)[::-1] # position of the hydrophones
dc =  0.15*npy.arange(0,64) # all the distance possible between the hydrophones



freq = 13000
nomf = str(int(freq/1000))+"kHz"
EDN = [ [ echants[k*50:(k+1)*50,:]  ,"rc1000e_m3_"+nomf+"_"+str(k)+".pick" ] for k in range(0,20) ] 
def calcul_rc(edn):
    echants_tmp = edn[0]
    nom = edn[1]
    RC = npy.zeros(len(echants_tmp))
    for kl,echant in enumerate(echants_tmp):
        MV = ModeleVertical(108,1523,echant[0],1000,echant[1],echant[2],echant[3],echant[4],echant[5],freq)
        MSA = ModeleSourceAntenne(MV,50,9000)
        RC[kl] = MSA.rayon_correlation_verticale(pcapteurs,dc)
    pickle.dump([echants_tmp,RC],open(nom,"wb"))

with Pool(20) as pool:
    pool.map( calcul_rc , EDN )
    
freq = 11000
nomf = str(int(freq/1000))+"kHz"
EDN = [ [ echants[k*50:(k+1)*50,:]  ,"rc1000e_m3_"+nomf+"_"+str(k)+".pick" ] for k in range(0,20) ]
def calcul_rc(edn):
    echants_tmp = edn[0]
    nom = edn[1]
    RC = npy.zeros(len(echants_tmp))
    for kl,echant in enumerate(echants_tmp):
        MV = ModeleVertical(108,1523,echant[0],1000,echant[1],echant[2],echant[3],echant[4],echant[5],freq)
        MSA = ModeleSourceAntenne(MV,50,9000)
        RC[kl] = MSA.rayon_correlation_verticale(pcapteurs,dc)
    pickle.dump([echants_tmp,RC],open(nom,"wb"))

with Pool(20) as pool:
    pool.map( calcul_rc , EDN )

freq = 5000
nomf = str(int(freq/1000))+"kHz"
EDN = [ [ echants[k*50:(k+1)*50,:]  ,"rc1000e_m3_"+nomf+"_"+str(k)+".pick" ] for k in range(0,20) ]
def calcul_rc(edn):
    echants_tmp = edn[0]
    nom = edn[1]
    RC = npy.zeros(len(echants_tmp))
    for kl,echant in enumerate(echants_tmp):
        MV = ModeleVertical(108,1523,echant[0],1000,echant[1],echant[2],echant[3],echant[4],echant[5],freq)
        MSA = ModeleSourceAntenne(MV,50,9000)
        RC[kl] = MSA.rayon_correlation_verticale(pcapteurs,dc)
    pickle.dump([echants_tmp,RC],open(nom,"wb"))

with Pool(20) as pool:
    pool.map( calcul_rc , EDN )


freq = 7000
nomf = str(int(freq/1000))+"kHz"
EDN = [ [ echants[k*50:(k+1)*50,:]  ,"rc1000e_m3_"+nomf+"_"+str(k)+".pick" ] for k in range(0,20) ]
def calcul_rc(edn):
    echants_tmp = edn[0]
    nom = edn[1]
    RC = npy.zeros(len(echants_tmp))
    for kl,echant in enumerate(echants_tmp):
        MV = ModeleVertical(108,1523,echant[0],1000,echant[1],echant[2],echant[3],echant[4],echant[5],freq)
        MSA = ModeleSourceAntenne(MV,50,9000)
        RC[kl] = MSA.rayon_correlation_verticale(pcapteurs,dc)
    pickle.dump([echants_tmp,RC],open(nom,"wb"))

with Pool(20) as pool:
    pool.map( calcul_rc , EDN )

    
