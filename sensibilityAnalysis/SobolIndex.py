#coding: utf-8

import pickle
import sys
sys.path.append("../")
import numpy as npy
import pandas as pds
from ChaosPolynomials import *
from matplotlib import pyplot as plt

ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(1640-1620) + 1620
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.59-0.59) + 0.59
echants[:,3] = echants[:,3]*(0.003-0.001) + 0.001
echants[:,4] = echants[:,4]*(50-10) + 10
echants[:,5] = echants[:,5]*(140-60) + 60
A = npy.array([ 1620 , 1650 , 0.59 , 0.001 , 10 , 60 ])
B = npy.array([ 1640 , 1750 , 1.59 , 0.003 , 50 , 140 ])
nomsParams = [ "vs" , "rhos" , "epsa" , "sigma" , "lv", "l" ]


def SobolIndex_computation(MVL,coeffs):
    D = npy.sum(coeffs[1:]**2)
    ESI = []
    ESTI = []
    for ki in range(0,6):
        print(ki)
        SI = 0.0
        STI = 0.0
        for kp,mpoly in enumerate(MVL.Polys):
            if (mpoly.degrees[ki] > 0 ) & (npy.sum(mpoly.degrees)==mpoly.degrees[ki]):
                SI += coeffs[kp]**2
            if mpoly.degrees[ki] >0:
                STI += coeffs[kp]**2
        SI /= D
        STI /= D
        ESI.append(SI[0])
        ESTI.append(STI[0])
    return ESI,ESTI

def SobolIndex_computation_main(freq):
    print(freq)
    #RC = pickle.load(open("donnees/rc"+ne+"_"+str(freq)+".pick","rb"))
    RC = pickle.load(open("rc"+ne+"_"+str(freq)+".pick","rb"))
    RC = npy.array([RC]).T

    MVL = MultivariateLegendre(A=A,B=B)
    coeffs = MVL.regression(echants,RC)
    ESI, ESTI = SobolIndex_computation(MVL,coeffs)
    dico = {"si":ESI,"sti":ESTI}
    df = pds.DataFrame(data=dico,index=nomsParams)
    df.to_csv("SI_"+str(freq)+".csv")

    
def lecture_resultats(freqs,nomsParams):
    fig = plt.figure(1,figsize=(16,9))
    AX = [ fig.add_subplot(1,len(freqs),k+1) for k in range(0,len(freqs)) ]
    print(len(AX))
    bar_width = 0.35
    index_bar = npy.arange(len(nomsParams))
    for kf,freq in enumerate(freqs):
        df = pds.read_csv("./donnees/SI_"+str(freq)+".csv",index_col=0,header=0)
        AX[kf].bar(index_bar,df["si"], bar_width, color='r',label='Sobol index')
        AX[kf].bar(index_bar + bar_width,df["sti"], bar_width, color='g',label='Total sobol index')
        AX[kf].set_xticks(index_bar+bar_width/2)
        AX[kf].set_xticklabels(nomsParams)
        AX[kf].legend()
    plt.show()

if __name__=="__main__":
    lecture_resultats([2000],( "vs" , "rhos" , "epsa" , "sigma" , "lv", "lh" ))
