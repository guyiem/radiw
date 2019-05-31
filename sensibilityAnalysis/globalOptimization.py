#coding: utf-8

from scipy import optimize
import numpy as npy
import numpy.matlib
import pickle
import pandas as pds
import sys
sys.path.append("../")
from ModeleVertical import *
from ChaosPolynomials import *

# global optimization using annealing

Freqs = [ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ]
pcapteurs = 50 - 0.015*npy.arange(0,640)[::-1] # position of the hydrophones
dc =  0.015*npy.arange(0,640) # all the distance possible between the hydrophones
rce = pickle.load(open("donnees/rce_sac2016.pick","rb"))

zf = 108
ve = 1523
rhoE = 1000
def fonctionOpt(tup):
    print(' valeurs en cours : ',tup)
    vs = tup[0]
    epsa = tup[1]
    sig = tup[2]
    lcv = tup[3]
    RC = npy.array([])
    for freq in Freqs:
        print(' \n freq = ',freq)
        MV = ModeleVertical(zf,ve,vs*100,1000,1700,epsa,sig/1000,10*lcv,100,freq,"exacte")
        MSA = ModeleSourceAntenne(MV,50,9000)
        rct = MSA.rayon_correlation_verticale(pcapteurs,dc)
        RC = npy.append(RC,rct)
    err = npy.linalg.norm(RC - RCE)
    print(" erreur = ",err)
    return err

res = optimize.dual_annealing( fonctionOpt, bounds=((15.5,16.5),(0.5,2.0),(0.005*1000,0.08*1000),(0.5,5.0)) , local_search_options={"method":"L-BFGS-B","options":{"gtol":0,"ftol":0.003}} )

res  = res['x']
res[0] = res[0]*100
res[2] = res[2]/1000
res[3] = res[3]*10
pickle.dump(res,open("annealing_result.pick","wb"))
print('\n erreur = ',fonctionOpt(res),'\n')

