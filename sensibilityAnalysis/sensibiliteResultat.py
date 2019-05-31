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
          'font.size' : 35,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
mpl.rc("xtick",labelsize=23)
mpl.rc("ytick",labelsize=23)

# TODO : convert some french to english. Transforming the code frome a script to something more "universal"

###########################################
# the purpose of this script is to conduct
# a posteriori analysis of the optimal
# correlation radius we found
###########################################


# number of saved metamodel we use
numModele = "11"


##########################
# LOADING DATAS
##########################
# loading of the experimentals radius, optimal theoritecal ones
rce = pickle.load(open("donnees/rce_sac2016.pick","rb")) # experimental radius
params_optims, rco = pickle.load(open("./donnees/rco_sac2016.pick","rb"))
rco = npy.array(rco)
print(params_optims)
print(" error of optimal correlations radius : ",npy.linalg.norm(rce-rco))
# end of loading of the experimentals radius, optimal theoritecal ones

# loading theoretical radius of the meta-model, and the meta-models
rc2000 = pickle.load(open("donnees/rc1000e_m"+str(numModele)+"_2kHz.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000e_m"+str(numModele)+"_5kHz.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000e_m"+str(numModele)+"_7kHz.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000e_m"+str(numModele)+"_9kHz.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000e_m"+str(numModele)+"_11kHz.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000e_m"+str(numModele)+"_13kHz.pick","rb"))
RCB = npy.array([ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ])
CPM = npy.array(pickle.load(open("donnees/ensModeles"+str(numModele)+".pick","rb")))
pmin = npy.array(CPM[0].A)
pmax = npy.array(CPM[0].B)
print(" limites MM ")
print(*pmin)
print(*pmax)
print( " params optims " )
print(params_optims,"\n")
# end of loading theoretical radius of the meta-model, and the meta-models

# choosing frequencies, subsampling if necessary
Freqs = npy.array([ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]) # all the frequencies we have meta-model
ifreq = [ 0 , 1 , 2 , 3 , 4 , 5  ] # frequencies we kept
Freqs = Freqs[ifreq]
print(Freqs)
RCB = RCB[ifreq]
CPM = CPM[ifreq]
rce = rce[ifreq]
rco = rco[ifreq]
# end of choice of frequencies
##########################
# LOADING DATAS
##########################


############################################
# SETTING THE QUADRATURES FOR THE COVARIANCE
############################################
# computing the residual
residus2 = 1/len(ifreq) * npy.sum(npy.abs(rce - rco)**2)

npq = 100_000 # number of points for MC method
print(" generating samples for MC method ")
echants = npy.zeros((npq,6))
VS_mc = npy.random.uniform(pmin[0],pmax[0],npq)
EPS_mc = npy.random.uniform(pmin[2],pmax[2],npq)
SIG_mc = npy.random.uniform(pmin[3],pmax[3],npq)
LV_mc = npy.random.uniform(pmin[4],pmax[4],npq)
Volume = (pmax[0]-pmin[0]) * ( pmax[2]-pmin[2]) * (pmax[3] - pmin[3] ) * (pmax[4] - pmin[4])
for k in range(0,npq):
    echants[k,:] = [VS_mc[k] , 1700 , EPS_mc[k] , SIG_mc[k] , LV_mc[k] , 100 ]
DB = 1/len(echants)
print(" end of generation of samples ")

print(" computing the weight for the quadrature ")
RCE = npy.matlib.repmat(rce,npq,1)
RC = npy.zeros((npq,len(ifreq)))
for km,cpm in enumerate(CPM):
    RC[:,km] = cpm.eval_model(echants.T)
poids = npy.exp(-npy.sum((RC - RCE)**2,axis=1)/(2*residus2))
print(" enf of computation for the quadrature weight ")

# computing the covariance matrix
Cov = npy.zeros((6,6))
for i in [0, 2, 3, 4]:
    print(i)
    int3 = npy.sum( echants[:,i]*poids )*DB
    for j in [0, 2, 3, 4]:
        int1 = npy.sum( echants[:,i]*echants[:,j]*poids )*DB        
        int2 = npy.sum( echants[:,j]*poids)*DB
        int4 = npy.sum( poids  )*DB
        Cov[i,j] = (int1*int4 - int2*int3)/(int4*int4)
# end of computation for the covariance matrix

# computation of the normalized covariance matrix
CC = npy.copy(Cov)
for ki in range(0,6):
    for kj in range(0,6):
        CC[ki,kj] /= npy.sqrt(Cov[ki,ki])*npy.sqrt(Cov[kj,kj])
print(" normalized covariance matrix : ")
print(CC,"\n")
# enf of computation of the normalized covariance matrix
        
params = [ r"$c_s$" , r"$\rho_s$" , r"$\alpha$" , r"$\sigma$" , r"$l_v$" , r"$l_h$" ]
print("\n uniform std : ")
vcu = ((pmax-pmin)**2)/12 
for k in range(0,6):
    print(params[k],npy.sqrt(vcu[k]))

print("\n model std : ")
for k in range(0,6):
    print(params[k],npy.sqrt(Cov[k,k]))


# ------------------------------------------------
# BEGINNING OF GRAPHICS FOR POSTERIORI DENSITIES
# ------------------------------------------------
fig = plt.figure(1,figsize=(12,12))
liste = [ [0,2] , [0,3] , [0,4] , [2,3] , [2,4] , [3,4] ] # list of pairs of parameters index
NF = [ 1 , 4 , 7 , 5 , 8 , 9 ] # list of subplot numbers
for kp,paire in enumerate(liste):
    print(paire)
    k1 = paire[0]
    k2 = paire[1]
    if k1 != k2:
        ax = fig.add_subplot(3,3,NF[kp])
        XX,YY = npy.mgrid[ pmin[k1]:pmax[k1]:101j , pmin[k2]:pmax[k2]:100j ] # structures for graphics plot
        positions = npy.vstack([XX.ravel(), YY.ravel()]) # list of points for computing densities
        noyau = stats.gaussian_kde( echants[:100000,[k1,k2]].T , weights=poids[:100000] , bw_method="silverman") # computation of density
        ZZ = npy.reshape(noyau(positions).T, XX.shape)
        ax.pcolormesh(XX,YY,ZZ)
        ax.plot(params_optims[k1],params_optims[k2],'ro')
        if (NF[kp] == 7) | (NF[kp] == 8) | (NF[kp] == 9):
            ax.set_xlabel(params[k1])
        if (NF[kp] == 7) | (NF[kp] == 1) | (NF[kp] == 4):        
            ax.set_ylabel(params[k2])
        if k2 == 2:
            plt.yticks([0.75,1.25]) # to avoid a graphical superposition
        fig.subplots_adjust(hspace=0.4,bottom=0.1,top=0.9,wspace=0.5,left=0.1,right=0.95)

plt.show()
#plt.savefig("MS_potmp2_m"+str(numModele)+".png",dpi=300)

    
# print(Cov,"\n")
# print(npy.sqrt(Cov),"\n")
# print(*npy.diag(npy.sqrt(Cov)))
# print(*npy.diag(Cov))


# WEIGHT STUDY
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
