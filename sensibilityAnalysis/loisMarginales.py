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
Freqs = npy.array([ 2000 , 5000 , 7000 , 9000 , 11000 , 13000])
# chargement des rayons expérimentaux, théoriques optimaux, et aléatoires
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))
rct = npy.array(pickle.load(open("rct_sac2016.pick","rb")))
rc2000 = pickle.load(open("donnees/rc1000_2000_9000.pick","rb"))
rc5000 = pickle.load(open("donnees/rc1000_5000_9000.pick","rb"))
rc7000 = pickle.load(open("donnees/rc1000_7000_9000.pick","rb"))
rc9000 = pickle.load(open("donnees/rc1000_9000_9000.pick","rb"))
rc11000 = pickle.load(open("donnees/rc1000_11000_9000.pick","rb"))
rc13000 = pickle.load(open("donnees/rc1000_13000_9000.pick","rb"))
RCB = npy.array([ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ])
# fin chargement des rayons expérimentaux, théoriques optimaux, et aléatoires
pmin = npy.array([ 1600 ,  0.5 , 0.0001 , 5.0  ])
pmax = npy.array([ 1650 , 1.5 , 0.005  , 50.0  ])
# chargement des méta-modèles
CPM = npy.array(pickle.load(open("toto.pick","rb")))

# choix des fréquences
ifreq = [ 0 , 1 , 2 , 3 , 4 , 5 ]
Freqs = Freqs[ifreq]
print(Freqs)
RCB = RCB[ifreq]
CPM = CPM[ifreq]
rce = rce[ifreq]
rct = rct[ifreq]
params_optims = npy.array([ 1630 , 1.09 , 0.002 , 30 ])
##########################
# fin chargement des données
##########################

residus2 = 1/len(ifreq) * npy.sum(npy.abs(rce - rct)**2)
VV,EE,SS,LL = npy.mgrid[ pmin[0]:pmax[0]:10j , pmin[1]:pmax[1]:11j , pmin[2]:pmax[2]:12j , pmin[3]:pmax[3]:13j ]
#positions = npy.vstack([ VV.ravel() , RR.ravel() , EE.ravel() , SS.ravel() , LL.ravel() , LH.ravel()  ]).T

V = npy.linspace(pmin[0],pmax[0],10)
E = npy.linspace(pmin[1],pmax[1],11)
S = npy.linspace(pmin[2],pmax[2],12)
L = npy.linspace(pmin[3],pmax[3],13)
positions = [ V , E , S , L  ]
#VV,EE,SS,LL = npy.meshgrid( V , E , S , L )
RR = 1700*npy.ones(EE.shape)
LH = 100*npy.ones(EE.shape)
iv = npy.argmin( (V-1630)**2 )
ie = npy.argmin( (E-1.09)**2 )
isig = npy.argmin( (S-0.002)**2 )
ilv = npy.argmin( (L-30)**2 )

RC = []
for cpm in CPM:
    RC.append( cpm.eval_model([VV, RR, EE, SS, LL, LH ]) )

erreurs = 0
for kr,rc in enumerate(RC):
    erreurs += (rc - rce[kr])**2
erreurs /= 2*residus2
poids = npy.exp( - erreurs )

fig = plt.figure(1)
compteur = 1
liste = [ [0,1] , [0,2] , [0,3] , [1,2] , [1,3] , [2,3] ]
lpoids = [ poids[:,:,isig,ilv] , poids[:,ie,:,ilv] , poids[:,ie,isig,:] , poids[iv,:,:,ilv] , poids[iv,:,isig,:] , poids[iv,ie,:,:] ]
NF = [ 1 , 2 , 3 , 5 , 6 , 9 ]
TF = [ 10j , 11j , 12j , 13j ]
for kp,paire in enumerate(liste):
    print(paire)
    k1 = paire[0]
    k2 = paire[1]
    toto = [0,1,2,3]
    toto.remove(k1)
    toto.remove(k2)
    print(toto)
    ax = fig.add_subplot(3,3,NF[kp])
    XX,YY = npy.mgrid[ pmin[k1]:pmax[k1]:TF[k1] , pmin[k2]:pmax[k2]:TF[k2] ]
    poids_tmp = npy.sum(npy.sum(poids,axis=toto[1]),axis=toto[0])
    #poids_tmp = lpoids[kp]
    #import ipdb ; ipdb.set_trace()
    ax.pcolor(XX,YY,poids_tmp)
    ax.plot(params_optims[k1],params_optims[k2],'ro')
    #ax.set_title(params[k1]+" , "+params[k2])
    fig.subplots_adjust(hspace=0.4,bottom=0.1,top=0.9,wspace=0.5,left=0.1,right=0.9)
    #ax.set_xlim([pmin[k1], pmax[k1]])
    #ax.set_ylim([pmin[k2], pmax[k2]])
    compteur += 1
plt.show()

# import ipdb ; ipdb.set_trace()

