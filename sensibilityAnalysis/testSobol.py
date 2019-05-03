#coding: utf-8

import pickle
import sys
sys.path.append("../")
import numpy as npy
import pandas as pds
from ChaosPolynomials import *

ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()

RC = npy.array([echants[:,0] + echants[:,1] + 2*echants[:,2] + 4*echants[:,3]]).T

MVL = MultivariateLegendre()
coeffs = MVL.regression(echants,RC)
print(*coeffs)
print(len(MVL.Polys))

D = npy.sum(coeffs[1:]**2)
ESI = []
ESTI = []
for ki in range(0,6):
    print("\n",ki)
    SI = 0.0
    STI = 0.0
    for kp,mpoly in enumerate(MVL.Polys):
        if (mpoly.degrees[ki] > 0 ) & (npy.sum(mpoly.degrees)==mpoly.degrees[ki]):
            print("debug : ",kp,"\n", *mpoly.degrees, coeffs[kp] )
            SI += coeffs[kp]**2
        if mpoly.degrees[ki] >0:
            STI += coeffs[kp]**2
    SI /= D
    STI /= D
    ESI.append(SI[0])
    ESTI.append(STI[0])
print(ESI)
print(ESTI)
print(D)
