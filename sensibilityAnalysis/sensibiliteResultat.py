#coding: utf-8

import numpy as npy
import numpy.matlib
import pickle
import pandas as pds
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib import pyplot as plt
import scipy.stats as stats
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#mpl.rc('text',usetex='True','fontsize':16)
params = {'text.usetex' : True,
          'font.size' : 25,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)


##########################
# chargement des données
##########################
Freqs = npy.array([ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ])
# chargement des rayons expérimentaux, théoriques optimaux, et aléatoires
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))
params_optims, rco = pickle.load(open("./donnees/rco_sac2016_1.pick","rb"))
rco = npy.array(rco)
rc2000 = pickle.load(open("donnees/rc1000e_m1_2kHz.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000e_m1_5kHz.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000e_m1_7kHz.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000e_m1_9kHz.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000e_m1_11kHz.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000e_m1_13kHz.pick","rb"))
RCB = npy.array([ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ])
# fin chargement des rayons expérimentaux, théoriques optimaux, et aléatoires
CPM = npy.array(pickle.load(open("donnees/ensModeles1.pick","rb")))
pmin = npy.array(CPM[0].A)
pmax = npy.array(CPM[0].B)
print(" limites MM ")
print(pmin)
print(pmax)
print( " params optims " )
print(params_optims,"\n")
# chargement des méta-modèles


# choix des fréquences
ifreq = [ 0 , 1 , 2 , 3 , 4 , 5  ]
Freqs = Freqs[ifreq]
print(Freqs)
RCB = RCB[ifreq]
CPM = CPM[ifreq]
rce = rce[ifreq]
rco = rco[ifreq]
##########################
# fin chargement des données
##########################


#################################
# mise en place de la quadrature
#################################
#calcul du sigma2
residus2 = 1/len(ifreq) * npy.sum(npy.abs(rce - rco)**2)
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


print(" calcul termes génériques pour la quadrature ")
RCE = npy.matlib.repmat(rce,npq,1)
RC = npy.zeros((npq,len(ifreq)))
for km,cpm in enumerate(CPM):
    RC[:,km] = cpm.eval_model(echants.T)
poids = npy.exp(-npy.sum((RC - RCE)**2,axis=1)/(2*residus2))
# étude du poids
# nparam = 4
# subtil = npy.ones(6)
# subtil[nparam] = 0
# # (npy.abs(echants[:,0] - 1630)<5)
# # & (npy.abs(echants[:,2] - 1.09)<0.1)
# # & (npy.abs(echants[:,3] - 0.002)< 0.0002)
# # & (npy.abs(echants[:,4] - 30)<1)
# ind = npy.where((subtil[0]*npy.abs(echants[:,0] - 1630)<5)  & (subtil[2]*npy.abs(echants[:,2] - 1.09)<0.1)  & (subtil[3]*npy.abs(echants[:,3] - 0.002)< 0.0002) & (subtil[4]*npy.abs(echants[:,4] - 30)<1) )[0]
# echants_tmp = echants[ind,:]
# poids_tmp = poids[ind]
# ind = npy.argsort(echants_tmp[:,nparam])
# echants_tmp = echants_tmp[ind,:]
# poids_tmp = poids_tmp[ind]
# plt.plot(echants_tmp[:,nparam],poids_tmp)
# plt.show()
# fin étude du poids
print(" fin calcul termes génériques pour la quadrature ")

Cov = npy.zeros((6,6))
for i in [0, 2, 3, 4]:
    print(i)
    int3 = npy.sum( echants[:,i]*poids )*DB
    for j in [0, 2, 3, 4]:
        int1 = npy.sum( echants[:,i]*echants[:,j]*poids )*DB        
        int2 = npy.sum( echants[:,j]*poids)*DB
        int4 = npy.sum( poids  )*DB
        Cov[i,j] = (int1*int4 - int2*int3)/(int4*int4)


CC = npy.copy(Cov)
for ki in range(0,6):
    for kj in range(0,6):
        CC[ki,kj] /= npy.sqrt(Cov[ki,ki])*npy.sqrt(Cov[kj,kj])

print(CC)
        
params = [ r"$v_s$" , r"$\rho_s$" , r"$\alpha$" , r"$\sigma$" , r"$l_v$" , r"$l_h$" ]
print("\n écart-type cas uniforme : ")
vcu = ((pmax-pmin)**2)/12 
for k in range(0,6):
    print(params[k],npy.sqrt(vcu[k]))


print("\n poids exponentiel : ")
for k in range(0,6):
    print(params[k],npy.sqrt(Cov[k,k]))


fig = plt.figure(1,figsize=(12,12))
compteur = 1
liste = [ [0,2] , [0,3] , [0,4] , [2,3] , [2,4] , [3,4] ]
NF = [ 1 , 2 , 3 , 5 , 6 , 9 ]
for kp,paire in enumerate(liste):
    print(paire)
    k1 = paire[0]
    k2 = paire[1]
    if k1 != k2:
        #X = npy.linspace(pmin[k1],pmax[k1],101)    
        #Y = npy.linspace(pmin[k2],pmax[k2],201)
        ax = fig.add_subplot(3,3,NF[kp])
        #Y = npy.linspace(pmin[k2],pmax[k2],200)
        #XX,YY = npy.meshgrid(X,Y)
        if k1 == 0:
            XX,YY = npy.mgrid[ pmin[k1]+25:pmax[k1]:101j , pmin[k2]:pmax[k2]:100j ]
        else:
            XX,YY = npy.mgrid[ pmin[k1]:pmax[k1]:101j , pmin[k2]:pmax[k2]:100j ] 
        positions = npy.vstack([XX.ravel(), YY.ravel()])
        noyau = stats.gaussian_kde( echants[:10000,[k1,k2]].T , weights=poids[:10000] , bw_method="silverman")
        ZZ = npy.reshape(noyau(positions).T, XX.shape)
        ax.pcolormesh(XX,YY,ZZ)#,extent=[pmin[k1], pmax[k1], pmin[k2], pmax[k2]])
        ax.plot(params_optims[k1],params_optims[k2],'ro')
        ax.set_title(params[k1]+" , "+params[k2])
        if k2 == 2:
            plt.yticks([0.75,1.25])
        fig.subplots_adjust(hspace=0.4,bottom=0.1,top=0.9,wspace=0.5,left=0.1,right=0.9)
        #ax.set_xlim([pmin[k1], pmax[k1]])
        #ax.set_ylim([pmin[k2], pmax[k2]])
    compteur += 1

plt.savefig("covMatrix.png",dpi=300)

    
# print(Cov,"\n")
# print(npy.sqrt(Cov),"\n")
# print(*npy.diag(npy.sqrt(Cov)))
# print(*npy.diag(Cov))
