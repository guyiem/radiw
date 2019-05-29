#!usr/bin/env python3.5
# coding: utf-8

from scipy import optimize
import numpy as npy
from scipy import linalg
from math import pi,e
from matplotlib import pyplot as plt
import pickle
import pandas as pds
import sys
sys.path.append('../')
from ModeleVertical import *

pcapteurs = 50 - 0.015*npy.arange(0,640)[::-1] # position of the hydrophones
dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones

RCE = pickle.load(open("donnees/rce_sac2016.pick","rb"))
RCE = npy.array([[*RCE]])
Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]

pmin = npy.array([ 1530 , 1650 , 0.5 , 0.0001 , 5 , 50 ] )
pmax = npy.array([ 1630 , 1750 , 2.0 , 0.005 , 50 , 150 ] )
ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(pmax[0]-pmin[0]) + pmin[0]
echants[:,1] = echants[:,1]*(pmax[1]-pmin[1]) + pmin[1]
echants[:,2] = echants[:,2]*(pmax[2]-pmin[2]) + pmin[2]
echants[:,3] = echants[:,3]*(pmax[3]-pmin[3]) + pmin[3]
echants[:,4] = echants[:,4]*(pmax[4]-pmin[4]) + pmin[4]
echants[:,5] = echants[:,5]*(pmax[5]-pmin[5]) + pmin[5]

nm = "11"
rc2000 = pickle.load(open("donnees/rc1000e_m"+nm+"_2kHz.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000e_m"+nm+"_5kHz.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000e_m"+nm+"_7kHz.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000e_m"+nm+"_9kHz.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000e_m"+nm+"_11kHz.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000e_m"+nm+"_13kHz.pick","rb"))
RCB = npy.array([ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ])

Err = npy.linalg.norm( RCB - RCE.T , axis = 0)
ind0 = npy.where(Err<0.1)[0]
ind1 = npy.where(Err<0.06)[0]
ind2 = npy.where(Err<0.05)[0]
ind3 = npy.where(Err<0.0475)[0]

for ind in ind1:
    print(*echants[ind,:],RCB[:,ind],Err[ind])
print("\n")
for ind in ind2:
    print(*echants[ind,:],Err[ind],RCB[:,ind])
print("\n")
for ind in ind3:
    print(*echants[ind,:],Err[ind],RCB[:,ind])
print("\n")
    
# plt.plot(ind0,Err[ind0],'r+')
# plt.plot(ind1,Err[ind1],'bo')
# plt.plot(ind2,Err[ind2],'co')
# plt.plot(ind3,Err[ind3],'mo')
# plt.show()
# #import ipdb ; ipdb.set_trace()

zf = 108
ve = 1523
rhoE = 1000
rhoS = 1700
lch = 100
def fonctionOpt(tup):
    print(' valeurs en cours : ',tup)
    vs = tup[0]
    epsa = tup[1]
    sig = tup[2]
    lcv = tup[3]
    RC = npy.array([])
    for freq in Freqs:
        print(' \n freq = ',freq)
        MV = ModeleVertical(zf,ve,vs*100,rhoE,1793,epsa,sig/1000,10*lcv,lch,freq,"exacte")
        MSA = ModeleSourceAntenne(MV,50,9000)
        rct = MSA.rayon_correlation_verticale(pcapteurs,dc)
        RC = npy.append(RC,rct)
    return npy.linalg.norm(RC - RCE)


for ki,ind in enumerate(ind3):
    res = optimize.minimize(fonctionOpt,(echants[ind,0]/100,
                                         echants[ind,2],
                                         echants[ind,3]*1000,
                                         echants[ind,4]/10),bounds=((15.71,16.25),(0.5,2.0),(0.002*1000,0.005*1000),(0.5,5)),method='L-BFGS-B',options={'gtol':0,'ftol':0.001})
    print(res)
    res = res['x']
    res[0] = res[0]*100
    res[2] = res[2]/1000
    res[3] = res[3]*10
    print(*res)
    pickle.dump(res,open("res3"+str(ki)+".bin","wb"))


for ki,ind in enumerate(ind2):
    res = optimize.minimize(fonctionOpt,(echants[ind,0]/100,
                                         echants[ind,2],
                                         echants[ind,3]*1000,
                                         echants[ind,4]/10),bounds=((15.71,16.25),(0.5,2.0),(0.002*1000,0.005*1000),(0.5,5)),method='L-BFGS-B',options={'gtol':0,'ftol':0.001})
    print(res)
    res = res['x']
    res[0] = res[0]*100
    res[2] = res[2]/1000
    res[3] = res[3]*10
    print(*res)
    pickle.dump(res,open("res2"+str(ki)+".bin","wb"))


for ki,ind in enumerate(ind1):
    res = optimize.minimize(fonctionOpt,(echants[ind,0]/100,
                                         echants[ind,2],
                                         echants[ind,3]*1000,
                                         echants[ind,4]/10),bounds=((15.71,16.25),(0.5,2.0),(0.002*1000,0.005*1000),(0.5,5)),method='L-BFGS-B',options={'gtol':0,'ftol':0.001})
    print(res)
    res = res['x']
    res[0] = res[0]*100
    res[2] = res[2]/1000
    res[3] = res[3]*10
    print(*res)
    pickle.dump(res,open("res1"+str(ki)+".bin","wb"))
    
