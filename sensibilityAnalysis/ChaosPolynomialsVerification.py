#coding: utf-8

from matplotlib import pyplot as plt
import numpy as npy
import pickle
import pandas as pds
from ChaosPolynomials import *


def orthonormality_check():
    a = 5
    b = 10
    Leg = Legendre3(a=a,b=b)
    Y = npy.linspace(a,b,1001)
    dy = Y[1] - Y[0]
    L1 = Leg.leg1(Y)
    L2 = Leg.leg2(Y)
    L3 = Leg.leg3(Y)
    print(" dy : ",dy)
    print(" normes :")
    print(npy.sum( (L1*L1)[1:]/(b-a))*dy,npy.sum( (L2*L2)[1:]/(b-a))*dy,npy.sum( (L3*L3)[1:]/(b-a))*dy)
    print(" PS : ")
    print(npy.sum( (L1*L2)[1:]/(b-a))*dy,npy.sum( (L1*L3)[1:]/(b-a))*dy,npy.sum( (L2*L3)[1:]/(b-a))*dy)    

    
def data_model_check(A,B,samples,RC):
    CPM = ChaosPolynomialsModel3(A,B,samples,RC)
    print(npy.max(RC - CPM.eval_model(samples.T)))

    
def regression_check(A,B,samples):
    RC = samples[:,0] + 2*samples[:,1] + samples[:,2]**3 + samples[:,3]**2
    CPM = ChaosPolynomialsModel3(A,B,samples,RC)
    print(npy.max(npy.abs(RC - CPM.eval_model(samples.T))))


def saved_datas_check():
    ne = "1000"
    echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
    echants = echants_pds.to_numpy()
    echants[:,0] = echants[:,0]*(1650-1600) + 1600
    echants[:,1] = echants[:,1]*(1750-1650) + 1650
    echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
    echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
    echants[:,4] = echants[:,4]*(50-5) + 5
    echants[:,5] = echants[:,5]*(150-50) + 50
    CPM = pickle.load(open("ensModeles.pick","rb"))
    rc2000 = pickle.load(open("donnees/rc1000_2000.pick","rb"))
    rc5000 = pickle.load(open("donnees/rc1000_5000.pick","rb"))
    rc7000 = pickle.load(open("donnees/rc1000_7000.pick","rb"))
    rc9000 = pickle.load(open("donnees/rc1000_9000.pick","rb"))
    rc11000 = pickle.load(open("donnees/rc1000_11000.pick","rb"))
    rc13000 = pickle.load(open("donnees/rc1000_13000.pick","rb"))
    RCB = [ rc2000 , rc5000 , rc7000 , rc9000 , rc11000 , rc13000 ]
    pmin = [ 1600 , 1650 , 0.5 , 0.0001 , 5.0  , 50.0 ]
    pmax = [ 1650 , 1750 , 1.5 , 0.005  , 50.0 , 150.0 ]
    for km,cpm in enumerate(CPM):
        print(npy.max(RCB[km]),npy.min(RCB[km]),npy.median(RCB[km]))
        erreurs = npy.abs(cpm.eval_model(echants.T)-RCB[km])/RCB[km]
        print(npy.mean(erreurs),npy.median(erreurs),npy.std(erreurs),npy.max(erreurs),npy.min(erreurs),"\n")



        
def metamodel_check():
    nomsFichier = [ "2kHz.png" , "5kHz.png" , "7kHz.png" , "9kHz.png" , "11kHz.png", "13kHz.png" ]
    nomsParam = [ "vs" , "rhos" , "attenuation" , "sigma" , "lv" , "lh" ]
    vs = 1630
    epsa = 1.09
    lv = 30
    lh = 100
    sigma = 0.002
    rhos = 1700
    CPM = pickle.load(open("ensModeles.pick","rb"))
    Np = 101
    X0 = npy.array([ vs , rhos , epsa , sigma , lv , lh ])
    for km,cpm in enumerate(CPM):
        fig = plt.figure(km+1,figsize=(4,24))
        for kp in range(0,6):
            print(kp)
            ax = fig.add_subplot(6,1,kp+1)
            X = npy.linspace(cpm.A[kp],cpm.B[kp],Np)            
            sortie = npy.zeros(Np)
            for kx in range(0,Np):
                Xtmp = npy.array([ vs , rhos , epsa , sigma , lv , lh ])
                Xtmp[kp] = X[kx]
                sortie[kx] = cpm.eval_model(Xtmp)
            ax.plot(X,sortie)
            ax.set_title(nomsParam[kp])
        fig.subplots_adjust(hspace=0.4,bottom=0.1,top=0.9,wspace=0.5,left=0.15,right=0.97)
        fig.savefig(nomsFichier[km])
    
        
if __name__=="__main__":
    saved_datas_check()
    #metamodel_check()
    #########################
    # test data_model
    #########################
    # ne = "1000"
    # echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
    # echants = echants_pds.to_numpy()
    # echants[:,0] = echants[:,0]*(1650-1600) + 1600
    # echants[:,1] = echants[:,1]*(1750-1650) + 1650
    # echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
    # echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
    # echants[:,4] = echants[:,4]*(50-5) + 5
    # echants[:,5] = echants[:,5]*(150-50) + 50
    # A = [ 1600 , 1650 , 0.5 , 0.0001 , 5 , 50 ]
    # B = [ 1650 , 1750 , 1.5 , 0.005 , 50 , 150 ]
    # RC = pickle.load(open("donnees/rc1000_2000.pick","rb"))
    # print(npy.max(RC),npy.min(RC),npy.median(RC))
    # data_model_check(A,B,echants,RC)
    #########################
    # end test data_model
    #########################

    ########################
    # test regression
    ########################
    # ne = "1000"
    # echants_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
    # echants = echants_pds.to_numpy()
    # echants[:,0] = echants[:,0]*(1650-1600) + 1600
    # echants[:,1] = echants[:,1]*(1750-1650) + 1650
    # echants[:,2] = echants[:,2]*(1.5-0.5) + 0.5
    # echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
    # echants[:,4] = echants[:,4]*(50-5) + 5
    # echants[:,5] = echants[:,5]*(150-50) + 50
    # A = [ 1600 , 1650 , 0.5 , 0.0001 , 5 , 50 ]
    # B = [ 1650 , 1750 , 1.5 , 0.005 , 50 , 150 ]
    # regression_check(A,B,echants)
    ########################
    # end test regression
    ########################
