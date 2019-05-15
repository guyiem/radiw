#coding: utf-8

import numpy as npy
import numpy.matlib
import pickle
import pandas as pds
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *

##########################
# chargement des données
##########################
Freqs = npy.array([ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ])
# chargement des rayons expérimentaux, théoriques optimaux, et aléatoires
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))
rct = npy.array(pickle.load(open("rct_sac2016.pick","rb")))
rc2000 = pickle.load(open("donnees/rc1000_2000.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000_5000.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000_7000.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000_9000.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000_11000.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000_13000.pick","rb"))
RCB = npy.array([ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ])
# fin chargement des rayons expérimentaux, théoriques optimaux, et aléatoires
pmin = [ 1600 , 1650 , 0.5 , 0.0001 , 5.0  , 50.0 ]
pmax = [ 1650 , 1750 , 1.5 , 0.005  , 50.0 , 150.0 ]
# chargement des méta-modèles
CPM = npy.array(pickle.load(open("ensModeles.pick","rb")))

# choix des fréquences
ifreq = [ 0 , 1 , 2 , 3 ]
Freqs = Freqs[ifreq]
print(Freqs)
RCB = RCB[ifreq]
CPM = CPM[ifreq]
rce = rce[ifreq]
rct = rct[ifreq]
##########################
# fin chargement des données
##########################

params_optims = npy.array([ 1630 , 1700 , 1.09 , 0.002 , 30 , 100 ])


#################################
# mise en place de la quadrature
#################################
#calcul du sigma2
residus2 = 1/len(ifreq) * npy.sum(npy.abs(rce - rct)**2)
npq = 100_000

print(" début génération échants ")
echants = npy.zeros((npq,6))
VS_mc = npy.random.uniform(pmin[0],pmax[0],npq)
EPS_mc = npy.random.uniform(pmin[2],pmax[2],npq)
SIG_mc = npy.random.uniform(pmin[3],pmax[3],npq)
LV_mc = npy.random.uniform(pmin[4],pmax[4],npq)
Volume = (pmax[0]-pmin[0]) * ( pmax[2]-pmin[2]) * (pmax[3] - pmin[3] ) * (pmax[4] - pmin[4])
for k in range(0,npq):
    echants[k,:] = [VS_mc[k] , 1700 , EPS_mc[k] , SIG_mc[k] , LV_mc[k] , 100 ]
DB = 1/len(echants)
#params_optims = npy.array([ 1630 , 1700 , 1.09 , 0.002 , 30 , 100 ])
#indoptim = npy.argmin(npy.sum((npy.abs(params_optims - echants)/params_optims)**2,axis=1))
print(" fin génération échants ")

RCE = npy.matlib.repmat(rce,npq,1)
Cov = npy.zeros((6,6))
print(" calcul termes génériques pour la quadrature ")
RC = npy.zeros((npq,len(ifreq)))
for km,cpm in enumerate(CPM):
    RC[:,km] = cpm.eval_model(echants.T)
poids = npy.exp(-npy.sum((RC - RCE)**2,axis=1)/(2*residus2))
# étude du poids
# nparam = 1
# ind = npy.argsort(echants,axis=0)
# toto = echants[ind[:,nparam]]
# plt.plot(toto[:,nparam],poids[ind[:,nparam]])
plt.plot(poids)
plt.show()
# fin étude du poids
print(" fin calcul termes génériques pour la quadrature ")

for i in [0, 2, 3, 4]:
    print(i)
    int3 = npy.sum( echants[:,i]*poids )*DB
    for j in [0, 2, 3, 4]:
        int1 = npy.sum( echants[:,i]*echants[:,j]*poids )*DB        
        int2 = npy.sum( echants[:,j]*poids)*DB
        int4 = npy.sum( poids  )*DB
        Cov[i,j] = (int1 - int2*int3)/int4

print(Cov,"\n")
print(npy.sqrt(Cov),"\n")
print(*npy.diag(npy.sqrt(Cov)))
print(*npy.diag(Cov))
import ipdb ; ipdb.set_trace()
