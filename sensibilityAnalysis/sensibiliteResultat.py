#coding: utf-8

import numpy as npy
import pickle
import pandas as pds
import sys
sys.path.append("../")
from ModeleVertical import *

Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))
cs = 1630
rhos = 1700
alpha = 1.09
sigma = 0.002
lv = 30
lh = 100

pcapteurs = 50 - 0.015*npy.arange(0,640)[::-1] # position of the hydrophones
dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones
rct = []
for freq in Freqs:
    MV = ModeleVertical(108,1523,vs,1000,rhos,alpha,sigma,lv,lh,freq)
    MSA = ModeleSourceAntenne(MV,50,9000)
    rct.append(MSA.rayon_correlation_verticale(pcapteurs,dc))



    
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(1600-1650) + 1600
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
echants[:,4] = echants[:,4]*(50-5) + 5
echants[:,5] = echants[:,5]*(150-50) + 50
import ipdb ; ipdb.set_trace()

# calcul du sigma2
# residus2 = 1/len(Freqs) * npy.sum(npy.abs(rce - rct)**2)

# pmin = [ 1600 , 1650 , 0.5 , 0.0001 , 5.0  , 50.0 ]
# pmax = [ 1650 , 1750 , 1.5 , 0.005  , 50.0 , 150 ]
# npq = 100

# Cov = npy.zeros((6,6))
# # for i in [0, 2, 3, 4]:
# #     for j in [0, 2, 3, 4]:
# #         sortie = []
# #         for ne in range(0,1000):
            
# #         densite = npy.exp(
        
