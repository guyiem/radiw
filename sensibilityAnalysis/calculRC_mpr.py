#coding: utf-8

import pickle
from multiprocessing import Pool
import subprocess as sbp
import sys
sys.path.append("../")
import numpy as npy
import pandas as pds
import ChaosPolynomials

from ModeleVertical import *

# ------------------------------------------------------------------------
# given a csv file with sampling of a 6D latin hyper cube on [0,1],
# rescale the hyper-cube and compute correlation radius according to these
# parameters and save its, in the main goal to compute meta-model
# probably a cleaner way to deal with multiprocessing that using pre-recording on disk, but lack of time and that does the job.
# ------------------------------------------------------------------------


# define the repertory to save the datas, the number of the models to save
repertory =  "./donnees"
model_number = ... 
# end of definition

# frequencies wich we're going to correlation radius
Freqs = [ 2000, 5000, 7000, 9000, 11000, 13000]
# en frequencies

# loading of the csv file : no header, just 6 columns of floats
ne = "1000"
samples_pds = pds.read_csv("donnees/lhcs"+str(ne)+".csv",header=None)
samples = samples_pds.to_numpy()
# end loading of the csv file

# definition of the bounds of the parameters for the models
# order of the parameters and units :
# 0) sound speed in the sediments : m/s
# 1) density of the sediments : km/m^3
# 2) attenuation : decibel by wavelength
# 3) amplitude of fluctuation of sound speed in water : relative
# 4) vertical correlation length of sound speed in water : meter
# 5) horizontal correlation length of sound speed in water : meter
pmin = [ 1630 , 1650 , 0.5 , 0.0001 , 5  , 50 ]
pmax = [ 1680 , 1750 , 1.5 , 0.005  , 50 , 150 ]
# end definition of the bounds of the parameters for the models

# rescaling of the samples
samples[:,0] = samples[:,0]*( pmax[0] - pmin[0] ) + pmin[0]
samples[:,1] = samples[:,1]*( pmax[1] - pmin[1] ) + pmin[1]
samples[:,2] = samples[:,2]*( pmax[2] - pmin[2] ) + pmin[2]
samples[:,3] = samples[:,3]*( pmax[3] - pmin[3] ) + pmin[3]
samples[:,4] = samples[:,4]*( pmax[4] - pmin[4] ) + pmin[4]
samples[:,5] = samples[:,5]*( pmax[5] - pmin[5] ) + pmin[5]
# end : rescaling of the samples

# definition of the hydrophone array
pcapteurs = 50 - 0.15*npy.arange(0,64)[::-1] # position of the hydrophones
dc =  0.15*npy.arange(0,64) # all the distance possible between the hydrophones
# enf of definition of the hydrophone array

# loop, to compute for each frequency the correlations radius for all samples. Parallelization inside the loop
for freq in Freqs:
    # creation of a repertory to save temporary correlation radius
    bool_rep = sbp.call(["mkdir", "m"+str(model_number)])
    if bool_rep == 1:
        raise ValueError(" the repertory already exists ")
    # end of creation of a repertory to save temporary correlation radius

    # sub-sampling samples for parallelization. Association to each subsamble of a name for saving radius.
    nomf = str(int(freq/1000))+"kHz"    
    EDN = [ [ samples[k*50:(k+1)*50,:]  ,"./m"+str(model_number)+"/rc"+str(ne)+"e_m"+str(model_number)+"_"+nomf+"_"+str(k)+".pick" ] for k in range(0,20) ]
    # end of sub-sampling
    
    # definition of the function for multiprocessing
    def calcul_rc(edn):
        samples_tmp = edn[0]
        nom = edn[1]
        RC = npy.zeros(len(samples_tmp))
        for kl,echant in enumerate(samples_tmp):
            MV = ModeleVertical(108,1523,echant[0],1000,echant[1],echant[2],echant[3],echant[4],echant[5],freq)
            MSA = ModeleSourceAntenne(MV,50,9000)
            RC[kl] = MSA.rayon_correlation_verticale(pcapteurs,dc)
        pickle.dump([samples_tmp,RC],open(nom,"wb"))
    # end of definition of the function for multiprocessing
    with Pool(20) as pool:
        pool.map( calcul_rc , EDN )


# loop on frequency to concatenate the subresult in one file by frequency
for freq in Freqs:
    ECH = npy.zeros((ne,6))
    RC = npy.zeros(ne)
    for k in range(0,20):
        etmp,rctmp = pickle.load(open("./m"+str(model_number)+"/rc"+str(ne)+"e_m"+str(model_number)+"_"+str(int(freq/1000))+"kHz_"+str(k)+".pick","rb"))
        ECH[k*50:(k+1)*50,:] = etmp # WARNING : sub-sampling is made here for 1000 samples
        RC[k*50:(k+1)*50] = rctmp # WARNING : sub-sampling is made here for 1000 samples
    erreur = npy.max(npy.abs(samples-ECH))
    if erreur >= 0:
        raise ValueError( " samples doesn't match ")    
    pickle.dump(RC,open(repertory."/rc"+str(ne)+"e_m"+str(model_number)+"_"+str(int(freq/1000))+"kHz.pick","wb"))

    
