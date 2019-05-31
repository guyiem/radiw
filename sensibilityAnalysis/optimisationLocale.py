#!usr/bin/env python3.5
# coding: utf-8

from scipy import optimize
import numpy as npy
from scipy import linalg
from math import pi,e
from matplotlib import pyplot as plt
import pickle
import sys
sys.path.append('../')
from ModeleVertical import *

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#mpl.rc('text',usetex='True','fontsize':16)
params = {'text.usetex' : True,
          'font.size' : 30,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

# loading the experimental correlation radius
RCE = pickle.load(open("donnees/rce_sac2016.pick","rb"))

pcapteurs =  50 - 0.015*npy.arange(0,640)[::-1]
dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones
Freqs = [2000,5000,7000,9000,11000,13000]

zf = 108
ve = 1523
rhoE = 1000
rhoS = 1700
lch = 100
lcv = 0.3

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
    err = npy.linalg.norm(RC - RCE)
    print(" erreur = ",err)
    return err

res = optimize.minimize(fonctionOpt,(16.0,1.5,0.003*1000,1.2),bounds=((15.8,16.2),(1.2,2.0),(0.0001*1000,0.005*1000),(0.5,5)),method='L-BFGS-B',options={'gtol':0,'ftol':0.001})
#res = optimize.minimize(fonctionOpt,(16.0,1.5,0.06),bounds=((15.9,16.2),(1.2,1.6),(0.05,0.08)),method='L-BFGS-B',options={'gtol':0,'ftol':0.1})
print(res)
res  = res['x']
print(res[0]*100,res[1],res[2]/1000,res[3]*10)


print('\n erreur = ',fonctionOpt(res),'\n')
