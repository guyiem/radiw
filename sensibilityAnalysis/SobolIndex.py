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

nomsParams = [ "cs" , "rhos" , "epsa" , "sigma" , "lv", "lh" ]

def SobolIndex_computation(cpm):
    """
    Given a ChaosPolynomialsModels3 object, compute Sobol index and total Sobol index

    input : cpm, a ChaosPolynomialsModels3 object
    output : ESI,ESTI : lists of Sobol indexes and total Sobol indexes for each parameter
    """
    D = npy.sum(cpm.Poids[1:]**2)
    ESI = []
    ESTI = []
    # loop on each parameter
    for ki in range(0,len(cpm.A)): 
        print(ki)
        SI = 0.0
        STI = 0.0
        # loop on the polynoms of the meta-model
        for kp,mpoly in enumerate(cpm.Polys):
            if (mpoly.degrees[ki] > 0 ) & (npy.sum(mpoly.degrees)==mpoly.degrees[ki]):
                SI += cpm.Poids[kp]**2
            if mpoly.degrees[ki] >0:
                STI += cpm.Poids[kp]**2
        SI /= D
        STI /= D
        ESI.append(SI)
        ESTI.append(STI)
    return ESI,ESTI

def SobolIndex_computation_main(CPM,Freqs,name):
    """
    Take a set of meta-models to given frequencies, compute Sobol and total Sobol index, and save them

    inputs : CPM, an iterative of ChaosPolynomialsModels3 object
            Freqs, an iterative of frequency (in Hertzà, each frequency corresponding to a meta-model of CPM
            name : a string, the "main name" to record files.
    outputs : none
    """
    for km,cpm in enumerate(CPM):
        ESI, ESTI = SobolIndex_computation(cpm)
        dico = {"si":ESI,"sti":ESTI}
        df = pds.DataFrame(data=dico,index=nomsParams)
        df.to_csv(name+"_"+str(int(Freqs[km]/1000))+"_kHz.csv") # we add to the main name the frequency number

    
def lecture_resultats(Freqs,nomsParams,name,savefig=True,namefig="SI.png"):
    """
    read and plot the Sobol and total Sobol index using bar plots.

    inputs : Freqs.  A iterative of frequencies. For each of its, we have save Sobol Index using SobolIndex_computation_main(...)
            nomsParams : a iterative of strings , corresponding to the names of the parameters. Can be given in latex.
            name : a string. The main name to load SobolIndex save with SobolIndex_computation_main(...)

            savefig, optional : a boolean. Set to true if we want to save the figure, false if we want to show()
            namgefig, optional : a string, the name to save the figure

    outputs : none
    """
    fig = plt.figure(1,figsize=(17.6,4.95))
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    AX = [ fig.add_subplot(1,len(Freqs),k+1) for k in range(0,len(Freqs)) ]
    bar_width = 0.35
    index_bar = npy.arange(len(nomsParams))
    for kf,freq in enumerate(Freqs):
        df = pds.read_csv(name+"_"+str(int(freq/1000))+"_kHz.csv",index_col=0,header=0)
        AX[kf].bar(index_bar,df["sti"], bar_width, color='g',label='Total Sobol index')
        AX[kf].bar(index_bar,df["si"], bar_width, color='r',label='Sobol index')
        AX[kf].set_xticks(index_bar)

        # following three lines : to trace the Sobol index and total Sobol index on two differents columns
        # AX[kf].bar(index_bar,df["si"], bar_width, color='r',label='Sobol index')
        # AX[kf].bar(index_bar + bar_width,df["sti"], bar_width, color='g',label='Total Sobol index')
        # AX[kf].set_xticks(index_bar+bar_width/2)
        AX[kf].set_xticklabels(nomsParams)
        AX[kf].set_ylim(0,0.65)
        AX[kf].set_title(str(int(freq/1000))+" kHz")
        #AX[kf].legend()
    #ax.legend(*AX[0].get_legend_handles_labels())
    fig.subplots_adjust(hspace=0.4,bottom=0.1,top=0.9,wspace=0.4,left=0.05,right=0.97)
    if savefig:
        plt.savefig(namefig)
    else:
        plt.show()

    
if __name__=="__main__":
    numModele = "11"
    ##########################
    # chargement des données
    ##########################
    Freqs = npy.array([ 2000 , 5000 , 7000 , 9000 , 11000 , 13000 ])
    CPM = npy.array(pickle.load(open("donnees/ensModeles"+str(numModele)+".pick","rb")))
    pmin = npy.array(CPM[0].A)
    pmax = npy.array(CPM[0].B)
    print(" limites MM ")
    print(*pmin)
    print(*pmax)
    # chargement des méta-modèles

    #SobolIndex_computation_main(CPM,Freqs)

    lecture_resultats([2000,5000,7000,9000,11000,13000],( r"$c_s$" , r"$\rho_s$" , r"$\alpha$" , r"$\sigma$" , r"$l_v$", r"$l_h$" ),"./donnees/SI",savefig=False)
    #SobolIndex_computation_main(CPM,Freqs)
