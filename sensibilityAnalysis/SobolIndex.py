#coding: utf-8

import pickle
import sys
sys.path.append("../")
import numpy as npy
import pandas as pds
from ChaosPolynomials import *
from matplotlib import pyplot as plt
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#mpl.rc('text',usetex='True','fontsize':16)
params = {'text.usetex' : True,
          'font.size' : 25,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

ne = "1000"
echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(1650-1600) + 1600
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
echants[:,4] = echants[:,4]*(50-5) + 5
echants[:,5] = echants[:,5]*(150-50) + 50
A = npy.array([ 1600 , 1650 , 0.5 , 0.0001 , 5 , 50 ])
B = npy.array([ 1650 , 1750 , 1.5 , 0.005 , 50 , 150 ])
nomsParams = [ "vs" , "rhos" , "epsa" , "sigma" , "lv", "lh" ]


def SobolIndex_computation(CPM,coeffs):
    D = npy.sum(coeffs[1:]**2)
    ESI = []
    ESTI = []
    for ki in range(0,6):
        print(ki)
        SI = 0.0
        STI = 0.0
        for kp,mpoly in enumerate(CPM.Polys):
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
    RC = pickle.load(open("donnees/rc"+ne+"_"+str(freq)+".pick","rb"))
    RC = npy.array([RC]).T

    CPM = ChaosPolynomialsModel3(A,B,echants,RC)
    coeffs = CPM.regression(echants,RC)
    ESI, ESTI = SobolIndex_computation(CPM,coeffs)
    dico = {"si":ESI,"sti":ESTI}
    df = pds.DataFrame(data=dico,index=nomsParams)
    df.to_csv("SI_"+str(freq)+".csv")

    
def lecture_resultats(freqs,nomsParams):
    fig = plt.figure(1,figsize=(17.6,4.95))
    #AX = [ fig.add_subplot(1,len(freqs),k+1) for k in range(0,len(freqs)) ]
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    AX = [ fig.add_subplot(1,len(freqs),k+1) for k in range(0,len(freqs)) ]
    bar_width = 0.35
    index_bar = npy.arange(len(nomsParams))
    for kf,freq in enumerate(freqs):
        df = pds.read_csv("./donnees/SI_"+str(freq)+".csv",index_col=0,header=0)
        AX[kf].bar(index_bar,df["sti"], bar_width, color='g',label='Total Sobol index')
        AX[kf].bar(index_bar,df["si"], bar_width, color='r',label='Sobol index')
        AX[kf].set_xticks(index_bar)
        # AX[kf].bar(index_bar,df["si"], bar_width, color='r',label='Sobol index')
        # AX[kf].bar(index_bar + bar_width,df["sti"], bar_width, color='g',label='Total Sobol index')
        # AX[kf].set_xticks(index_bar+bar_width/2)
        AX[kf].set_xticklabels(nomsParams)
        AX[kf].set_ylim(0,0.65)
        AX[kf].set_title(str(int(freq/1000))+" kHz")
        #AX[kf].legend()
    #ax.legend(*AX[0].get_legend_handles_labels())
    fig.subplots_adjust(hspace=0.4,bottom=0.1,top=0.9,wspace=0.4,left=0.05,right=0.97)
    plt.savefig("toto.png")

if __name__=="__main__":
    lecture_resultats([2000,5000,7000,9000,11000,13000],( r"$v_s$" , r"$\rho_s$" , r"$\alpha$" , r"$\sigma$" , r"$l_v$", r"$l_h$" ))
    #SobolIndex_computation_main(13000)
