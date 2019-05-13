#coding: utf-8

import numpy as npy
import pickle
import pandas as pds
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *

Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))
# vs = 1630
# rhos = 1700
# alpha = 1.09
# sigma = 0.002
# lv = 30
# lh = 100

# pcapteurs = 50 - 0.015*npy.arange(0,640)[::-1] # position of the hydrophones
# dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones
# rct = []
# for freq in Freqs:
#     MV = ModeleVertical(108,1523,vs,1000,rhos,alpha,sigma,lv,lh,freq)
#     MSA = ModeleSourceAntenne(MV,50,9000)
#     rct.append(MSA.rayon_correlation_verticale(pcapteurs,dc))
# pickle.dump(rct,open("rct_sac2016.pick","wb"))

rct = pickle.load(open("rct_sac2016.pick","rb"))

# echants_pds = pds.read_csv("donnees/lhcs1000.csv",header=None)
# echants = echants_pds.to_numpy()
# echants[:,0] = echants[:,0]*(1650-1600) + 1600
# echants[:,1] = echants[:,1]*(1750-1650) + 1650
# echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
# echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
# echants[:,4] = echants[:,4]*(50-5) + 5
# echants[:,5] = echants[:,5]*(150-50) + 50
pmin = [ 1600 , 1650 , 0.5 , 0.0001 , 5.0  , 50.0 ]
pmax = [ 1650 , 1750 , 1.5 , 0.005  , 50.0 , 150 ]

rc2000 = pickle.load(open("donnees/rc1000_2000.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000_5000.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000_7000.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000_9000.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000_11000.pick","rb"))
rc13000 = rc11000

RCB = [ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ]

#calcul du sigma2
residus2 = 1/len(Freqs) * npy.sum(npy.abs(rce - rct)**2)

# CPM = []
# for kf,freq in enumerate(Freqs):
#     CPM.append( ChaosPolynomialsModel3(pmin,pmax,echants,RCB[kf]) )
# pickle.dump(CPM,open("ensModeles.pick","wb"))
CPM = pickle.load(open("ensModeles.pick","rb"))

# mise en place de la quadrature
# VS = npy.arange(pmin[0],pmax[0],0.1)
# EPSA = npy.arange(pmin[2],pmax[2],0.1)
# SIGMA = npy.arange(pmin[3],pmax[3],0.0001)
# LV = npy.arange(pmin[4],pmax[4],0.1)
# DB = (VS[1]-VS[0]) * (EPSA[1]-EPSA[0]) * (SIGMA[1]-SIGMA[0]) * (LV[1]-LV[0] )
VS = npy.linspace(pmin[0],pmax[0],100)
EPSA = npy.linspace(pmin[2],pmax[2],100)
SIGMA = npy.linspace(pmin[3],pmax[3],100)
LV = npy.linspace(pmin[4],pmax[4],100)
DB = (VS[1]-VS[0]) * (EPSA[1]-EPSA[0]) * (SIGMA[1]-SIGMA[0]) * (LV[1]-LV[0] )

# construction des échantillons
rhos = 1700
lh = 100
echants = []
print(" début construction échants ")
for vs in VS:
    for epsa in EPSA:
        for sigma in SIGMA:
            for lv in LV:
                echants.append(npy.array([vs,rhos,epsa,sigma,lv,lh]))

Cov = npy.zeros((6,6))
for i in [0, 2, 3, 4]:
    print(i)
    for j in [0, 2, 3, 4]:
        int1 = 0
        int2 = 0
        int3 = 0
        int4 = 0
        for ech_tmp in echants:
            RC = []
            for km,cpm in enumerate(CPM):
                RC.append(cpm.eval_model(ech_tmp[0]))
                RC = npy.array(RC)
            poids = npy.exp(-npy.sum((rce - RC)**2)/(2*residus2))
            int1 += ech_tmp[i]*ech_tmp[j]*poids
            int2 += ech_tmp[j]*poids
            int3 += ech_tmp[i]*poids
            int4 += poids
        int1 *= DB
        int2 *= DB
        int3 *= DB
        int4 *= DB
        Cov[i,j] = (int1 - int2*int3)/int4

print(npy.sqrt(Cov),"\n")
print(*npy.diag(npy.sqrt(Cov)))
