#coding: utf-8

import numpy as npy
import pickle
import pandas as pds

ne = "1000"
echants_pds = pds.read_csv("../donnees/lhcs"+str(ne)+".csv",header=None)
echants = echants_pds.to_numpy()
echants[:,0] = echants[:,0]*(1630-1530) + 1530
echants[:,1] = echants[:,1]*(1750-1650) + 1650
echants[:,2] = echants[:,2]*(2.0-0.5) + 0.5
echants[:,3] = echants[:,3]*(0.005-0.0001) + 0.0001
echants[:,4] = echants[:,4]*(50-5) + 5
echants[:,5] = echants[:,5]*(150-50) + 50

for freq in [2000 , 5000, 7000, 9000, 11000, 13000]:
    ECH = npy.zeros((1000,6))
    RC = npy.zeros(1000)
    for k in range(0,20):
        etmp,rctmp = pickle.load(open("rc1000e_m11_"+str(int(freq/1000))+"kHz_"+str(k)+".pick","rb"))
        ECH[k*50:(k+1)*50,:] = etmp
        RC[k*50:(k+1)*50] = rctmp
    print(npy.max(npy.abs(echants-ECH)))
    # print(npy.min(ECH[:,0]),npy.max(ECH[:,0]))
    # print(npy.min(ECH[:,1]),npy.max(ECH[:,1]))
    # print(npy.min(ECH[:,2]),npy.max(ECH[:,2]))
    # print(npy.min(ECH[:,3]),npy.max(ECH[:,3]))
    # print(npy.min(ECH[:,4]),npy.max(ECH[:,4]))
    # print(npy.min(ECH[:,5]),npy.max(ECH[:,5]),"\n")
    pickle.dump(RC,open("../donnees/rc1000e_m11_"+str(int(freq/1000))+"kHz.pick","wb"))
