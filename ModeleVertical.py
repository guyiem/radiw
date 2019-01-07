# coding: utf-8

"""
code used in the paper "acoustic and geoacoustic inverse problems in randomly perturbed shallow water waveguide environments". Contains 3 classes :

- ModeleVerticalDeterministe : modelize an homogeneous waveguide (no attenuation in the bottom, no fluctuation in the water. cf. section 3 of the paper)

- ModeleVertical : modelize a random and dissipative waveguide. Inherit form ModeleVerticalDeterministe, with addition of fluctuation in water, and attenuation in sediment.

- ModeleSourceAntenne : modelize a waveguide with a source and an hydrophones array

"""


import numpy as npy
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as slinalg
from math import pi,e
from matplotlib import pyplot as plt

import cProfile
import pstats
import yappi

npy.set_printoptions(linewidth=100,precision=2)

class ModeleVerticalDeterministe(object):
    """
    modelize an homogeneous waveguide (no attenuation in the bottom, no fluctuation in the water. cf. section 3 of the paper).

    Attr :
    - zf : depth of the waveguide in meters ( positive float )
    - ve : sound speed in water in m/s (positive float)
    - vs : sound speed in sediment in m/s (positive float)
    - rhoe : density of water in kg/m^3 (positive float)
    - rhos : density of sediment in kg/m^3 (positive float)
    - freq : frequency of our Helmholtz problem. In Hertz (positive float)
    - choix_methode : ununsed actually. string to choose the computation method, between and exact and an approximation.

    Main methods :
     - calcul_valeurs_propres_propag() : compute the transverse wave numbers
    """
    
    def __init__(self,zf,ve,vs,rhoe,rhos,freq,choix_methode="exacte"):
        self.zf = zf
        self.ve = ve
        self.vs = vs
        self.rhoe = rhoe
        self.rhos = rhos
        self.choix_methode = choix_methode
        self.freq = freq        
        # calcul nombres d'onde
        self.wne,self.wns,self.wne2,self.wns2 = self.waveNumber(freq)
        self.Kej = self.calcul_valeurs_propres_propag()
        self.Kxj = npy.sqrt( self.wne2 - self.Kej**2 )
        self.Kxj2 = self.Kxj * self.Kxj
        if npy.min(self.Kxj2)<self.wns2 or npy.max(self.Kxj2)>self.wne2:
            raise ValueError( " pb pour les Kxj dans le calcul propagatif ")
        self.Sigmaj = self.zf*npy.sqrt( self.wne2 - self.Kxj2 )
        self.Zetaj = self.zf*npy.sqrt( self.Kxj2 - self.wns2 )
        self.Nm = len(self.Kxj)
        # fin calcul nombres d'onde
        self.Aj = self.AjP()

    def waveNumber(self,freq):
        omega = 2*pi*freq
        wne = omega/self.ve
        wns = omega/self.vs
        wne2 = wne*wne
        wns2 = wns*wns
        return wne,wns,wne2,wns2
    
    # ---------------------------------
    # début calcul des valeurs propres
    # ---------------------------------
    def fct_zeros_propag(self,Kez): # vectorialisé en Kez
        return self.rhos/self.rhoe * Kez * npy.cos( Kez * self.zf ) + npy.sqrt( self.wne2 - self.wns2 - Kez**2 ) * npy.sin( Kez * self.zf)

    
    def deriv_fct_zeros_propag(self,Kez): # vectorialisé en Kez
        racine = npy.sqrt( self.wne2 - self.wns2 - Kez**2 )
        t1 = self.rhos/self.rhoe*( npy.cos( Kez * self.zf ) - self.zf * Kez * npy.sin( Kez * self.zf ) )
        t2 = Kez/racine * npy.sin( Kez * self.zf ) + self.zf * racine * npy.cos( Kez * self.zf )
        return t1 + t2
    
    def calcul_valeurs_propres_propag(self):
        dk = npy.sqrt(self.wne2-self.wns2)/(self.freq/10*self.zf)
        K = npy.arange(0,npy.sqrt(self.wne2-self.wns2),dk)
        tolN = pow(10,-12)
        Y = self.fct_zeros_propag(K)
        iK = npy.where(npy.abs(Y)<0.1)[0]
        sortie = []
        for ind in iK:
            ktmp = K[ind]
            compteur = 0
            err = 1.0
            while err>=tolN and compteur<10:
                k0 = ktmp
                ktmp = ktmp - self.fct_zeros_propag(ktmp)/self.deriv_fct_zeros_propag(ktmp)
                err = npy.abs(self.fct_zeros_propag(ktmp)) + npy.abs(k0-ktmp)
                compteur += 1
            sortie.append(ktmp)
        sortie = npy.round(npy.array(sortie),decimals=5)
        sortie = npy.unique(sortie)
        ind = npy.where( (sortie>0.0000000000001) & (sortie<npy.sqrt(self.wne2-self.wns2)-0.000000000001) )[0]        
        return sortie[ind]
    # ---------------------------------
    # début calcul des valeurs propres
    # ---------------------------------


    # -----------------------------------------------------------------------------
    # calcul des coefficients et fonctions pour les modes propagatifs et radiatifs
    # -----------------------------------------------------------------------------
    def AjP(self): # vectorialisé en Kxj
        t1 = 1/self.rhoe * ( 1 - npy.sin(2*self.Sigmaj)/(2*self.Sigmaj) ) + 1/self.rhos * npy.sin(self.Sigmaj)**2/self.Zetaj
        return npy.sqrt( (2/self.zf)/t1 )

    
    def AgR(self,Gamma): # vectorialisé en Gamma
        etaG = self.zf * npy.sqrt( self.wne2 - Gamma )
        xhiG = self.zf * npy.sqrt( self.wns2 - Gamma )
        num = xhiG * self.rhos * self.zf
        d1 = (xhiG * npy.sin(etaG))**2
        d2 = ( self.rhos/self.rhoe * etaG * npy.cos(etaG) )**2
        return npy.sqrt( num/(pi*(d1+d2)) )

    
    def modesPropagatifs(self,z):
        assert (z < self.zf), ("le point de calcul est en dessous du fond")
        sortie = self.Aj * npy.sin( self.Sigmaj * z / self.zf )
        return sortie
    
    def modesRadiatifs(self,Gamma,Z):
        if npy.max(self.Gamma>wns2):
            raise ValueError( " il y a un gamma trop grand pour les modes radiatifs ")
        sortie = npy.zeros((Z.size,Gamma.size)) + 0j
        # calcul des coeffs
        etaG = self.zf * npy.sqrt( self.wne2 - Gamma )
        xhiG = self.zf * npy.sqrt( self.wns2 - Gamma )
        Ag = self.AgR(Gamma) 
        # fin calcul des coeffs
        indEau = npy.where(Z<=self.zf)[0]
        indSed = npy.where(Z>self.zf)[0]
        for kg,gam in enumerate(Gamma):
            sortie[indEau,kg] = Ag[kg] * npy.sin( etaG[kg] * Z[indEau]/self.zf )
            sortie[indSed,kg] = Ag[kg] * ( npy.sin(etaG[kg]) * npy.cos(xhiG[kg]*(Z[indSed]-self.zf)/self.zf) + self.rhos/self.rhoe * etaG[kg]/xhiG[kg]*npy.cos(etaG[kg]) * npy.sin(xhiG[kg]*(Z[indSed]-self.zf)/self.zf) )
        return sortie
    # ----------------------------------------------------------------------------------
    # fin  calcul des coefficients et fonctions pour les modes propagatifs et radiatifs
    # ----------------------------------------------------------------------------------

    # ----------------------------------
    # ajout d'une source
    # ----------------------------------
    def aj0(self,Ffreq,zs):
        return npy.sqrt(self.Kxj) * 0.5* Ffreq * self.modesPropagatifs(zs)

    def ag0p(self,Ffreq,zs,Gamma):
        if npy.max(Gamma>self.wns2) or npy.min(Gamma<0):
            raise ValueError( " Gamma trop grand ou trop petit ")
        return 0.5 * Ffreq * self.modesRadiatifs(Gamma,zs)

    def ag0n(self,Ffreq,zs,Gamma):
        if npy.max(Gamma>0):
            raise ValueError(" il y a un Gamma positif ")
        return -0.5 * Ffreq * self.modesRadiatifs(Gamma,zs)

    def ag0(self,Ffreq,zs,Gamma):
        indN = npy.where(Gamma<=0)[0]
        indP = npy.where( (Gamma>=0) and (Gamma<=self.wns2) )[0]
        GN = Gamma[indN]
        GP = Gamma[indP]
        sortie = npy.zeros(Gamma.shape)
        sortie[indN] = self_.ag0n(Ffreq,zs,GN)
        sortie[indP] = self.ag0p(Ffreq,zs,GP)
        return sortie
    # ----------------------------------
    # fin ajout de la source
    # ----------------------------------

    
    def fonctionS(self,P):
        lvp = self.lv*P
        lvp2 = lvp*lvp
        constante = self.lv/(2*P*( lvp2 + 1)**2)
        sortie = ( 4*lvp*npy.exp(-self.zf/self.lv) * (
            npy.cos(P*self.zf) - lvp*npy.sin(P*self.zf) )
            + (lvp2 + 1)*( npy.sin(2*P*self.zf)
            - lvp*npy.cos(2*P*self.zf)
            + 2*P*self.zf)
            + lvp*(lvp2 -3))
        return constante*sortie


# ---------------------------------------------
# modèle aléatoire absorbant
# ---------------------------------------------
class ModeleVertical(ModeleVerticalDeterministe):

    def __init__(self,zf,ve,vs,rhoe,rhos,cadb,sigma,lv,lh,freq,choix_methode="exacte"):
        ModeleVerticalDeterministe.__init__(self,zf,ve,vs,rhoe,rhos,freq,choix_methode)
        # paramètre pour l'absorption et l'aléatoire
        #self.ca = cadb/(40*pi*npy.log(e))
        self.ca = cadb/(20*pi*npy.log10(e))
        print("vs : ",vs,'\n',4*vs/(4+self.ca**2),2*vs*self.ca/(4+self.ca**2))
        self.sigma = sigma
        self.lv = lv

        self.lv8 = lv**8
        self.lv7 = lv**7
        self.lv6 = lv**6
        self.lv5 = lv**5
        self.lv4 = lv**4
        self.lv3 = lv**3
        self.lv2 = lv**2
        
        self.lh = lh
        print(" ca : ",self.ca)
        print(" début calcul des gamma ")
        self.Gamma = self.Gamma_jl()
        print(" fin calcul des gamma , début calcul des lambda2 ")
        #self.Gamma = npy.array([[-2,0.5,1.5],[0.5,-3,2.5],[1.5,2.5,-4]]) #self.Gamma_jl()
        self.Lambda2 = self.Lambda2_j()
        print(" fin calcul des lambda2, début calcul des lambda1 ")
        self.Lambda1 = npy.array([ self.Lambda1_j(ij,kxj) for ij,kxj in enumerate(self.Kxj) ])
        print(" fin calcul des lambda1 ")
        #self.Lambda2 = npy.zeros(self.Nm)
        #self.Lambda1 = npy.zeros(self.Nm) #npy.array([ self.Lambda1_j(ij,kxj) for ij,kxj in enumerate(self.Kxj) ])1
        print( " Lambda1 : ",npy.min(self.Lambda1),npy.max(self.Lambda1),npy.mean(self.Lambda1),npy.std(self.Lambda1))
        print( " Lambda2 : ",npy.min(self.Lambda2),npy.max(self.Lambda2),npy.mean(self.Lambda2),npy.std(self.Lambda2))
        print(" Gamma : ",npy.min(self.Gamma),npy.max(self.Gamma),npy.mean(self.Gamma),npy.std(self.Gamma))


        
    def I_jl(self):        
        aj,al = npy.meshgrid(self.Kej,self.Kej)

        lv = self.lv
        zf = self.zf
        sin = npy.sin
        cos = npy.cos
        exp = npy.exp

        Sigmaj,Sigmal = npy.meshgrid(self.Sigmaj,self.Sigmaj)
        Zetaj,Zetal = npy.meshgrid(self.Zetaj,self.Zetaj)
        Aj,Al = npy.meshgrid(self.Aj,self.Aj)

        if self.choix_methode=="approchee":
            return (Aj*Aj*Al*Al)/4 * ( self.fonctionS( Sigmaj/self.zf - Sigmal/self.zf ) + self.fonctionS( Sigmaj/self.zf + Sigmal/self.zf ) )

        if self.choix_methode=="approchee2":
            return (Aj*Aj*Al*Al)/4 * ( self.fonctionS( Sigmaj/self.zf - Sigmal/self.zf ) )
        
        if self.choix_methode=="exacte":
            return Al*Al * Aj*Aj *  (1.0/npy.power((al*al)*(lv*lv)*2.0+(al*al*al*al)*(lv*lv*lv*lv)+(aj*aj)*(lv*lv)*2.0+(aj*aj*aj*aj)*(lv*lv*lv*lv)-(al*al)*(aj*aj)*(lv*lv*lv*lv)*2.0+1.0,2.0)*((al*al*al)*lv*sin(aj*zf*2.0)*(1.0/2.0)-(aj*aj*aj)*lv*sin(al*zf*2.0)*(1.0/2.0)-al*(aj*aj*aj)*(lv*lv)*2.0+(al*al*al)*aj*(lv*lv)*2.0-al*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv)*4.0+(al*al*al*al*al)*aj*(lv*lv*lv*lv)*4.0-al*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*2.0+(al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv)*2.0+(al*al*al*al*al)*(lv*lv*lv)*sin(aj*zf*2.0)*(3.0/2.0)-(aj*aj*aj*aj*aj)*(lv*lv*lv)*sin(al*zf*2.0)*(3.0/2.0)+(al*al*al*al*al*al*al)*(lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*(3.0/2.0)-(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*sin(al*zf*2.0)*(3.0/2.0)+(al*al*al*al*al*al*al*al*al)*(lv*lv*lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*(1.0/2.0)-(aj*aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(al*zf*2.0)*(1.0/2.0)-(al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*10+(al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*10-(al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*zf+(al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv)*zf-(al*al*al)*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*zf*2.0+(al*al*al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*zf*2.0+al*(aj*aj*aj)*(lv*lv)*npy.power(cos(al*zf),2.0)*2.0-(al*al*al)*aj*(lv*lv)*npy.power(cos(al*zf),2.0)*2.0+al*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*4.0-(al*al*al*al*al)*aj*(lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*6.0+al*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*2.0-(al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*6.0-(al*al*al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*2.0+al*(aj*aj*aj)*(lv*lv)*npy.power(cos(aj*zf),2.0)*2.0-(al*al*al)*aj*(lv*lv)*npy.power(cos(aj*zf),2.0)*2.0+al*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*6.0-(al*al*al*al*al)*aj*(lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*4.0+al*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*6.0-(al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*2.0+al*(aj*aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*2.0+(al*al)*(aj*aj*aj)*(lv*lv*lv)*sin(al*zf*2.0)*(5.0/2.0)+(al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*sin(al*zf*2.0)*4.0+(al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv)*sin(al*zf*2.0)*(5.0/2.0)+(al*al)*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(al*zf*2.0)*(5.0/2.0)-(al*al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(al*zf*2.0)*(5.0/2.0)-(al*al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(al*zf*2.0)*(1.0/2.0)-(al*al*al)*(aj*aj)*(lv*lv*lv)*sin(aj*zf*2.0)*(5.0/2.0)-(al*al*al)*(aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*(5.0/2.0)-(al*al*al*al*al)*(aj*aj)*(lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*4.0+(al*al*al)*(aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*(1.0/2.0)+(al*al*al*al*al)*(aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*(5.0/2.0)-(al*al*al*al*al*al*al)*(aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*(5.0/2.0)+al*(aj*aj*aj)*lv*zf-(al*al*al)*aj*lv*zf+(al*al)*aj*lv*sin(al*zf*2.0)-al*(aj*aj)*lv*sin(aj*zf*2.0)+(al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*2.0-(al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*2.0+(al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*6.0+(al*al*al)*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*2.0-(al*al*al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*6.0+(al*al*al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*6.0-(al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*2.0-(al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*6.0+(al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*2.0-(al*al*al)*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*6.0+(al*al*al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*6.0-(al*al*al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(aj*zf),2.0)*2.0+al*(aj*aj*aj*aj*aj)*(lv*lv*lv)*zf*3.0-(al*al*al*al*al)*aj*(lv*lv*lv)*zf*3.0+al*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*zf*3.0-(al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv)*zf*3.0+al*(aj*aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*zf-(al*al*al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv*lv)*zf+(al*al*al*al)*aj*(lv*lv*lv)*sin(al*zf*2.0)*3.0+(al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv)*sin(al*zf*2.0)*3.0+(al*al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv*lv)*sin(al*zf*2.0)-al*(aj*aj*aj*aj)*(lv*lv*lv)*sin(aj*zf*2.0)*3.0-al*(aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*sin(aj*zf*2.0)*3.0-al*(aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*sin(aj*zf*2.0)-al*(aj*aj*aj)*(lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*2.0+(al*al*al)*aj*(lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*2.0-al*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*6.0+(al*al*al*al*al)*aj*(lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*6.0-al*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*6.0+(al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*6.0-al*(aj*aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*2.0+(al*al*al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*2.0+(al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*2.0-(al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*2.0+(al*al*al)*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*4.0-(al*al*al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*npy.power(cos(aj*zf),2.0)*4.0-(al*al)*aj*lv*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*2.0+al*(aj*aj)*lv*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*2.0-(al*al*al*al)*aj*(lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*6.0-(al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*6.0-(al*al*al*al*al*al*al*al)*aj*(lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*2.0+al*(aj*aj*aj*aj)*(lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*6.0+al*(aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*6.0+al*(aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*2.0-(al*al)*(aj*aj*aj)*(lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*10-(al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*14-(al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*12-(al*al)*(aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*6.0+(al*al*al*al)*(aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*10-(al*al*al*al*al*al)*(aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*npy.power(cos(aj*zf),2.0)*sin(al*zf)*2.0+(al*al*al)*(aj*aj)*(lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*10+(al*al*al)*(aj*aj*aj*aj)*(lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*12+(al*al*al*al*al)*(aj*aj)*(lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*14+(al*al*al)*(aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*2.0-(al*al*al*al*al)*(aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*10+(al*al*al*al*al*al*al)*(aj*aj)*(lv*lv*lv*lv*lv*lv*lv)*npy.power(cos(al*zf),2.0)*cos(aj*zf)*sin(aj*zf)*6.0-(al*al)*(aj*aj*aj*aj)*(lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*4.0+(al*al*al*al)*(aj*aj)*(lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*4.0-(al*al)*(aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*8.0+(al*al*al*al*al*al)*(aj*aj)*(lv*lv*lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*8.0-(al*al)*(aj*aj*aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*4.0+(al*al*al*al)*(aj*aj*aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*12-(al*al*al*al*al*al)*(aj*aj*aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*12+(al*al*al*al*al*al*al*al)*(aj*aj)*(lv*lv*lv*lv*lv*lv*lv*lv)*cos(al*zf)*cos(aj*zf)*sin(al*zf)*sin(aj*zf)*4.0)*(-1.0/2.0))/(al*aj*(al*al-aj*aj))+al*aj*(lv*lv*lv*lv)*exp(-zf/lv)*1.0/npy.power((al*al)*(lv*lv)*2.0+(al*al*al*al)*(lv*lv*lv*lv)+(aj*aj)*(lv*lv)*2.0+(aj*aj*aj*aj)*(lv*lv*lv*lv)-(al*al)*(aj*aj)*(lv*lv*lv*lv)*2.0+1.0,2.0)*(sin(al*zf)*sin(aj*zf)+(al*al*al)*(lv*lv*lv)*cos(al*zf)*sin(aj*zf)+(aj*aj*aj)*(lv*lv*lv)*cos(aj*zf)*sin(al*zf)+(al*al)*(lv*lv)*sin(al*zf)*sin(aj*zf)+(aj*aj)*(lv*lv)*sin(al*zf)*sin(aj*zf)+al*lv*cos(al*zf)*sin(aj*zf)+aj*lv*cos(aj*zf)*sin(al*zf)+al*aj*(lv*lv)*cos(al*zf)*cos(aj*zf)*2.0-al*(aj*aj)*(lv*lv*lv)*cos(al*zf)*sin(aj*zf)-(al*al)*aj*(lv*lv*lv)*cos(aj*zf)*sin(al*zf))*4.0

    
    def I_jg(self,ij,kxj,gamma):
        #if npy.max(gamma)>self.wns2:
        #    raise ValueError( " gamma trop grand ")
        
        ag = npy.sqrt(self.wne2 - gamma)
        aj = self.Kej[ij] #npy.sqrt( self.wne2 - kxj*kxj)

        lv = self.lv
        zf = self.zf
        sin = npy.sin
        cos = npy.cos
        exp = npy.exp

        etaG = self.zf * npy.sqrt( self.wne2 - gamma )
        xhiG = self.zf * npy.sqrt( self.wns2 - gamma )
        Ag = self.AgR(gamma)

        sigmaj = self.Sigmaj[ij]
        zetaj = self.Zetaj[ij]
        Aj = self.Aj[ij]

        ag2 = ag * ag
        aj2 = aj * aj
        ag3 = ag * ag2
        aj3 = aj * aj2
        ag4 = ag * ag3
        aj4 = aj * aj3
        ag5 = ag * ag4
        aj5 = aj * aj4
        ag6 = ag * ag5
        aj6 = aj * aj5
        ag7 = ag * ag6
        aj7 = aj * aj6
        ag8 = ag * ag7
        aj8 = aj * aj7
        ag9 = ag * ag8
        aj9 = aj * aj8

        agaj = ag*aj
        agaj2 = ag*aj2
        agaj3lv2 = ag*aj3*self.lv2
        agaj7 = ag*aj7
        agaj9 = ag*aj9
        ag3aj = ag3*aj
        ag3aj5 = ag3*aj5
        ag3aj7 = ag3*aj7
        agaj5 = ag*aj5
        ag2aj = ag2*aj
        ag5aj3 = ag5*aj3
        ag5aj = ag5*aj
        ag7aj = ag7*aj
        ag7aj3 = ag7*aj3
        ag9aj = ag9*aj
                
        cagzf = cos(ag*zf)        
        cagzf2 = cagzf**2
        cajzf = cos(aj*zf)
        cajzf2 = cajzf**2
        ccgj2 = cagzf*cajzf2
        ccg2j = cagzf2*cajzf
        ccg2j2 = cagzf2*cajzf2
        
        saj2zf = sin(aj*zf*2.0)
        sag2zf = sin(ag*zf*2.0)
        sajzf = sin(aj*zf)
        sagzf = sin(ag*zf)
        ssgj = sagzf*sajzf

        csgj = cagzf*sajzf

        csjg = cajzf*sagzf
        ccssgjgj = cagzf*csjg*sajzf

        ccsgj2g = ccgj2*sagzf
        ccsg2jj = ccg2j*sajzf

        ip2 = 1.0/((ag2*self.lv2*2.0+ag4*self.lv4+aj2*self.lv2*2.0+aj4*self.lv4-ag2*aj2*self.lv4*2.0+1.0)**2)
        #if self.choix_methode=="approchee":
        #    return (Ag*Ag*Aj*Aj)/4 * ( self.fonctionS( sigmaj/self.zf - etaG/self.zf ) + self.fonctionS( sigmaj/zf + etaG/self.zf ) )

        #if self.choix_methode=="approchee2":
        #    return (Ag*Ag*Aj*Aj)/4 * ( self.fonctionS( sigmaj/self.zf - etaG/self.zf ) )
        
        #if self.choix_methode=="exacte":
        
        return Ag*Ag * Aj*Aj * (ip2*( ag3*lv*saj2zf*0.5 - aj3*lv*sag2zf*0.5 - agaj3lv2*2.0 + ag3aj*self.lv2*2.0 - agaj5*self.lv4*4.0 + ag5aj*self.lv4*4.0 - agaj7*self.lv6*2.0 + ag7aj*self.lv6*2.0 + ag5*self.lv3*saj2zf*1.5 - aj5*self.lv3*sag2zf*1.5 + ag7*self.lv5*saj2zf*1.5 - aj7*self.lv5*sag2zf*1.5 + ag9*self.lv7*saj2zf*0.5 - aj9*self.lv7*sag2zf*0.5 - ag3aj5*self.lv6*10 + ag5aj3*self.lv6*10 - ag3aj5*self.lv5*zf + ag5aj3*self.lv5*zf - ag3aj7*self.lv7*zf*2.0 + ag7aj3*self.lv7*zf*2.0 + agaj3lv2*cagzf2*2.0 - ag3aj*self.lv2*cagzf2*2.0 + agaj5*self.lv4*cagzf2*4.0 - ag5aj*self.lv4*cagzf2*6.0 + agaj7*self.lv6*cagzf2*2.0 - ag7aj*self.lv6*cagzf2*6.0 - ag9aj*self.lv8*cagzf2*2.0 + agaj3lv2*cajzf2*2.0 - ag3aj*self.lv2*cajzf2*2.0 + agaj5*self.lv4*cajzf2*6.0 - ag5aj*self.lv4*cajzf2*4.0 + agaj7*self.lv6*cajzf2*6.0 - ag7aj*self.lv6*cajzf2*2.0 + agaj9*self.lv8*cajzf2*2.0 + ag2*aj3*self.lv3*sag2zf*2.5 + ag2*aj5*self.lv5*sag2zf*4.0 + ag4*aj3*self.lv5*sag2zf*2.5 + ag2*aj7*self.lv7*sag2zf*2.5 - ag4*aj5*self.lv7*sag2zf*2.5 - ag6*aj3*self.lv7*sag2zf*0.5 - ag3*aj2*self.lv3*saj2zf*2.5 - ag3*aj4*self.lv5*saj2zf*2.5 - ag5*aj2*self.lv5*saj2zf*4.0 + ag3*aj6*self.lv7*saj2zf*0.5 + ag5*aj4*self.lv7*saj2zf*2.5 - ag7*aj2*self.lv7*saj2zf*2.5 + ag*aj3*self.lv*zf - ag3aj*lv*zf + ag2aj*lv*sag2zf - agaj2*lv*saj2zf + ag3*aj3*self.lv4*cagzf2*2.0 - ag3aj5*self.lv6*cagzf2*2.0 + ag5aj3*self.lv6*cagzf2*6.0 + ag3aj7*self.lv8*cagzf2*2.0 - ag5*aj5*self.lv8*cagzf2*6.0 + ag7aj3*self.lv8*cagzf2*6.0 - ag3*aj3*self.lv4*cajzf2*2.0 - ag3aj5*self.lv6*cajzf2*6.0 + ag5aj3*self.lv6*cajzf2*2.0 - ag3aj7*self.lv8*cajzf2*6.0 + ag5*aj5*self.lv8*cajzf2*6.0 - ag7aj3*self.lv8*cajzf2*2.0 + agaj5*self.lv3*zf*3.0 - ag5aj*self.lv3*zf*3.0 + agaj7*self.lv5*zf*3.0 - ag7aj*self.lv5*zf*3.0 + agaj9*self.lv7*zf - ag9aj*self.lv7*zf + ag4*aj*self.lv3*sag2zf*3.0 + ag6*aj*self.lv5*sag2zf*3.0 + ag8*aj*self.lv7*sag2zf - ag*aj4*self.lv3*saj2zf*3.0 - ag*aj6*self.lv5*saj2zf*3.0 - ag*aj8*self.lv7*saj2zf - agaj3lv2*ccg2j2*2.0 + ag3aj*self.lv2*ccg2j2*2.0 - agaj5*self.lv4*ccg2j2*6.0 + ag5aj*self.lv4*ccg2j2*6.0 - agaj7*self.lv6*ccg2j2*6.0 + ag7aj*self.lv6*ccg2j2*6.0 - agaj9*self.lv8*ccg2j2*2.0 + ag9aj*self.lv8*ccg2j2*2.0 + ag3aj5*self.lv6*ccg2j2*2.0 - ag5aj3*self.lv6*ccg2j2*2.0 + ag3aj7*self.lv8*ccg2j2*4.0 - ag7aj3*self.lv8*ccg2j2*4.0 - ag2aj*lv*ccsgj2g*2.0 + agaj2*lv*ccsg2jj*2.0 - ag4*aj*self.lv3*ccsgj2g*6.0 - ag6*aj*self.lv5*ccsgj2g*6.0 - ag8*aj*self.lv7*ccsgj2g*2.0 + ag*aj4*self.lv3*ccsg2jj*6.0 + ag*aj6*self.lv5*ccsg2jj*6.0 + ag*aj8*self.lv7*ccsg2jj*2.0 - ag2*aj3*self.lv3*ccsgj2g*10 - ag2*aj5*self.lv5*ccsgj2g*14 - ag4*aj3*self.lv5*ccsgj2g*12 - ag2*aj7*self.lv7*ccsgj2g*6.0 + ag4*aj5*self.lv7*ccsgj2g*10 - ag6*aj3*self.lv7*ccsgj2g*2.0 + ag3*aj2*self.lv3*ccsg2jj*10 + ag3*aj4*self.lv5*ccsg2jj*12 + ag5*aj2*self.lv5*ccsg2jj*14 + ag3*aj6*self.lv7*ccsg2jj*2.0 - ag5*aj4*self.lv7*ccsg2jj*10 + ag7*aj2*self.lv7*ccsg2jj*6.0 - ag2*aj4*self.lv4*ccssgjgj*4.0 + ag4*aj2*self.lv4*ccssgjgj*4.0 - ag2*aj6*self.lv6*ccssgjgj*8.0 + ag6*aj2*self.lv6*ccssgjgj*8.0 - ag2*aj8*self.lv8*ccssgjgj*4.0 + ag4*aj6*self.lv8*ccssgjgj*12 - ag6*aj4*self.lv8*ccssgjgj*12 + ag8*aj2*self.lv8*ccssgjgj*4.0)*(-0.5))/(agaj*(ag2 - aj2)) + agaj*self.lv4*exp(-zf/lv)*ip2*(ssgj + ag3*self.lv3*csgj + aj3*self.lv3*csjg + ag2*self.lv2*ssgj + aj2*self.lv2*ssgj + ag*lv*csgj + aj*lv*csjg + agaj*self.lv2*cagzf*cajzf*2.0 - agaj2*self.lv3*csgj - ag2aj*self.lv3*csjg)*4.0

        

    def Lambda1_j(self,ij,kxj):
        omega2 = (2*pi*self.freq)**2
        # calcul du premier terme
        err1 = 1
        err2 = 1
        np = self.wns*1001
        compteur = 0
        while (err1>0.01) | (err2>0.01):
            Eta = npy.linspace(0,self.wns,int(np))
            deltaE = Eta[1]-Eta[0]
            Eta2 = Eta**2
            Integrande = []
            kej = self.Kej[ij]#npy.sqrt( self.wne2 - kxj*kxj)
            KeE = npy.sqrt( self.wne2 - Eta2 )
            epm = (Eta[1:]+Eta[:-1])/2
            epm2 = epm*epm
            Integrande = 1/( 1 + ((Eta-kxj)*self.lh)**2) * self.I_jg(ij,kxj,Eta2)
            intpm = 1/( 1 + ((epm-kxj)*self.lh)**2) * self.I_jg(ij,kxj,epm2)                                        
            ipm = npy.sum( intpm )*deltaE
            irg = npy.sum( Integrande[:-1] )*deltaE
            ird = npy.sum( Integrande[1:])*deltaE
            err1 = npy.abs(ipm-irg)/npy.abs(ipm)
            err2 = npy.abs(ipm-ird)/npy.abs(ipm)
            np = 2*np
            compteur += 1
        #print( "Lambda1_j, nb itération pour l'intégrale, erreur : ", compteur, err1 , err2)
        return (self.sigma**2 * omega2**2 * self.lh )/( 2 * self.rhoe**2 * self.ve**4  * kxj ) * ipm

    
    def Lambda2_j(self):
        omega2 = (2*pi*self.freq)**2
        # calcul du deuxième terme
        return self.ca * omega2 * ((self.Aj*npy.sin(self.Sigmaj))**2)*self.zf / ( self.Kxj * self.vs**2 * self.rhos * 2*self.Zetaj )
        
        
    def Gamma_jl(self):
        kxj,kxl = npy.meshgrid(self.Kxj,self.Kxj)
        Mtmp = self.sigma**2 * self.wne**4 * self.lh / ( 2* self.rhoe**2 * kxj * kxl * ( 1 + ((kxl-kxj)*self.lh)**2) ) * self.I_jl()
        npy.fill_diagonal(Mtmp,0)
        for ij in range(0,self.Nm):
            Mtmp[ij,ij] = -npy.sum(Mtmp[ij,:])
        return Mtmp
        # if choix_methode=="approchee": # /!\ tout à reprendre
        #     Aj,Al = npy.meshgrid(self.Aj,self.Aj)
        #     kej = npy.sqrt( wne2 - kxj*kxj)
        #     kel = npy.sqrt( wne2 - kxl*kxl)
        #     return ( self.sigma * wne2 * wne2 * self.lh /(8*self.rhoe*self.rhoe)
        #              * 1/(kxj*kxl)
        #              * 1/( 1 + ((kxl-kxj)*self.lh)**2)
        #              * Aj * Aj * Al * Al
        #              * ( self.fonctionS(kej-kel) + self.fonctionS(kej+kel)))
        # elif choix_methode=="exacte":
        #     return ( (2*pi*freq)**4/(2*kxj*kxl) ) * (self.sigma**2) * self.I_jl(freq,kxj,kxl)* ( self.lh/(1+ ((kxl-kxj)*self.lh)**2 )) / (self.rhoe**2 * self.ve**4 )
        # else:
        #     raise ValueError( " choix de methode inexistant ")


    def matriceA(self):
        MatriceA = npy.copy(self.Gamma)                        
        for ij,kxj in enumerate(self.Kxj):
            MatriceA[ij,ij] += -( self.Lambda1[ij] + self.Lambda2[ij]  )
        return MatriceA

    
    def matriceB(self):
        GamPla = self.Gamma.flatten()
        Lambda = self.Lambda1 + self.Lambda2
        Nm = self.Nm
        Mtmp = npy.copy(self.Gamma)
        npy.fill_diagonal(Mtmp,0)
        # construction B1 à B3
        B1 = sparse.diags(-2*GamPla)
        diagUn = npy.ones(Nm)
        B2 = sparse.diags(npy.array([ -lbd*diagUn for lbd in Lambda ]).flatten())
        B3 = sparse.diags(npy.array([ -Lambda for k in range(0,Nm) ]).flatten())
        # fin construction B1 à B3
        # construction B4
        B4 = sparse.block_diag([Mtmp for k in range(0,Nm)])
        # fin construction B4
        # construction B5
        diagG = npy.diag(self.Gamma) 
        B5 = sparse.diags( npy.array([ diagG for k in range(0,Nm)]).flatten() )
        # fin construction B5
        # construction B6
        liste  = [ [ sparse.diags(self.Gamma[kj,kl]*diagUn) for kj in range(0,Nm) ] for kl in range(0,Nm) ]
        for kj in range(0,Nm):
            liste[kj][kj] = None
        B6 = sparse.bmat(liste)
        # fin construction B6
        # début construction B7
        B7 = sparse.diags(npy.array([ s*diagUn for s in diagG ]).flatten())
        # fin construction B7
        B = B1 + B2 + B3 + B4 + B5 + B6 + B7
        B = B.tocsc()
        Btmp = B.copy()
        num_ligne = npy.arange(0,Nm)*Nm + npy.arange(0,Nm)
        for knl,nl in enumerate(num_ligne):
            indtmp = npy.where(B.indices==nl)[0]
            B.data[indtmp] = 0            
        # construction de la matrice pour les indices jj
        B = B.tocoo()
        diagjj = - 2*Lambda + 2*diagG # car la diag de Gamma, c'est - la somme des termes non-diag
        Mtmp = 4*Mtmp
        npy.fill_diagonal(Mtmp,diagjj)
        Bjj = sparse.block_diag([ sparse.csr_matrix((ligneTmp,(nl*npy.ones(Nm),npy.arange(0,Nm))),shape=(Nm,Nm)) for nl,ligneTmp in enumerate(Mtmp) ])        
        B += Bjj
        B.eliminate_zeros()
        B = B.tocsc()
        #import ipdb ; ipdb.set_trace()
        return B

        
    

        
class ModeleSourceAntenne(object):

    def __init__(self,MVA,zs,xa):
        self.MV = MVA
        self.zs = zs
        self.xa = xa
        self.aj0 = MVA.aj0(1,zs)
        print(" début du calcul des moments d'ordre 2")
        self.mo2 = self.moments_ordre2()
        print(" fin du calcul des moments d'ordre 2")
        assert (len(self.mo2) == len(self.MV.Kxj)), (" pb sur les moments d'ordre 2")

    def moments_ordre2(self):
        # construction de la matrice
        MatriceA = self.MV.matriceA()
        expM = linalg.expm(self.xa*MatriceA)        
        # fin construction de la matrice
        # calcul des espérances
        aj02 = self.aj0**2
        espe = npy.dot(expM,aj02)
        print('corre_verticale, espe :',npy.min(espe),npy.max(espe))
        print('corre_verticale, expM :',npy.min(expM),npy.max(expM))
        return espe
        #return npy.sum( espe/self.MV.Kxj * (self.modesPropagatifs(zs))**2 )

    
    def corre_verticale(self,pcapteurs,dcapteurs):
        # calcul de la corré verticale
        zM = pcapteurs[-1]
        zm = pcapteurs[0]
        sortie = npy.zeros(dcapteurs.shape)
        Aj2 = self.MV.Aj**2
        for ic,dc in enumerate(dcapteurs):
            #print(' dc = ',dc)
            sortie[ic] = npy.sum(0.5/(zM-zm-dc) * self.mo2 / self.MV.Kxj * Aj2 * ( npy.cos(self.MV.Kej*dc)*(zM-zm-dc) - 0.5/self.MV.Kej*( npy.sin(self.MV.Kej*(2*zM-dc)) - npy.sin(self.MV.Kej*(2*zm+dc)))) )
        return sortie



    def indice_scintillation(self,pc1d):
        assert len(pc1d.shape) == 1 , " les capteurs ne sont pas 1d "
        print(" début IS ")
        # calcul de E[ |a_j|^2 |a_l|^2 ]
        print(" Nm : ",self.MV.Nm)
        print(" début calcul matrice B ")
        matB = self.MV.matriceB()
        print(" fin calcul matrice B ")
        A0,AA0 = npy.meshgrid(self.aj0,self.aj0)
        P0 = (npy.abs(A0**2)*npy.abs(AA0**2)).flatten()
        print(" début calcul mo4 ")
        mo4 = slinalg.expm_multiply(self.xa*matB,P0)
        print(" fin calcul mo4 ")
        # fin calcul de E[ |a_j|^2 |a_l|^2 ]

        # calcul de E[I]^2
        print( " début calcul E[I]^2 " )
        phiJ2 = 0
        for pc in pc1d:
            phiJ2 += self.MV.modesPropagatifs(pc)**2
        espI2 = npy.sum( 1/self.MV.Kxj * phiJ2 * self.mo2 )**2
        print( " fin calcul E[I]^2 " )
        #espI2 = npy.sum( self.mo2 )**2 
        # fin calcul de E[I]^2L

        # calcul de E[ I^2 ]
        print( " début calcul E[I^2] " )
        BJ,BBJ = npy.meshgrid(npy.abs(self.MV.Kxj),npy.abs(self.MV.Kxj))
        BJL = (BJ*BBJ).flatten()
        PJ,PPJ = npy.meshgrid(phiJ2,phiJ2)
        termesNonCroisees = npy.eye(PJ.shape[0]).flatten()
        PJL2 = (PJ*PPJ).flatten()
        EI2 = 2* (1/BJL)*PJL2 * mo4 - termesNonCroisees*(1/BJL)*PJL2 * mo4
        return (npy.sum(EI2) - espI2) / espI2



def supp_diag(M):
    if M.shape[0] != M.shape[1]:
        raise ValueError(" la matrice n'est pas carré ")
    else:
        m = M.shape[0]
        Mtmp = npy.zeros((m,m-1))
        for i in range(0,m):
            ligne = M[i,:]
            ligne = npy.delete(ligne,i)
            Mtmp[i,:] = ligne
        return Mtmp
    

    
if __name__=="__main__":
    #MVA = ModeleVertical(104,1519,1540,1000,1500,0.5,0.001,40,100,"approchee")
    #MVA2 = ModeleVertical(104,1519,1540,1000,1500,0.5,0.001,40,100,"approchee2")
    freq = 1000
    yappi.start()
    MVE = ModeleVertical(104,1519,1540,1000,1500,0.0,0.001,40,100,freq,"exacte")
    MSA = ModeleSourceAntenne(MVE,50,10000)
    # -----------------
    # test matrice B
    # -----------------
    MSA.indice_scintillation()
    func_stats = yappi.get_func_stats()
    func_stats.save('callgrind.out', 'CALLGRIND')
    yappi.stop()
    yappi.clear_stats()

    # -------------------
    # fin test matrice B
    # -------------------
    
    
    # # -----------------------
    # # test lbd et gamma
    # # -----------------------
    # LBD1 = []
    # LBD2 = []
    # MatA = []
    # for MV in EMV:
    #     wne,wns,wne2,wns2 = MV.waveNumber(freq)
    #     Kej = MV.calcul_valeurs_propres_propag(freq)[:-5]
    #     Kxj = npy.sqrt( wne2 - Kej**2 )
    #     Lbd1 = npy.array([ MV.Lambda1_j(freq,kxj) for kxj in Kxj ])
    #     Lbd2 = npy.array([ MV.Lambda2_j(freq,kxj) for kxj in Kxj ])
    #     Lbd = Lbd1 + Lbd2
    #     Gam = npy.array([ [MV.Gamma_lj(freq,kxj,kxl) for kxj in Kxj] for kxl in Kxj ])
    #     Gam = supp_diag(Gam)
    #     #Gam = npy.round(Gam,9)
    #     LBD1.append(Lbd1)
    #     LBD2.append(Lbd2)
    #     MatA.append(MV.matriceA(freq))
    #     #print(Lbd)
    #     #print(Gam)
    #     print(" wne : ",wne)
    #     print(" lbd1 : ",npy.min(Lbd1),npy.max(Lbd1),npy.mean(Lbd1),npy.std(Lbd1))
    #     print(" lbd2 : ",npy.min(Lbd2),npy.max(Lbd2),npy.mean(Lbd2),npy.std(Lbd2))
    #     print(" gam : ",npy.min(Gam),npy.max(Gam),npy.mean(Gam),npy.std(Gam))
    #     print(" Mat A : ",npy.min(npy.abs(MatA)),npy.max(npy.abs(MatA)),npy.mean(npy.abs(MatA)),npy.std(npy.abs(MatA)))
    #     #print( " expA : ",linalg.expm(matriceA*10000))    
    #     #Betaj = npy.sqrt( wne2 - (pi*(npy.arange(0,len(Kej)) - 0.5)/104)**2 )

    # print('\n')
    # for lbd11 in LBD1:
    #     for lbd12 in LBD1:
    #         print(" err : ",npy.max(npy.abs(lbd11-lbd12)/npy.abs(lbd11)))

    # print('\n')
    # for lbd21 in LBD2:
    #     for lbd22 in LBD2:
    #         print(" err : ",npy.max(npy.abs(lbd21-lbd22)/npy.abs(lbd21)))
    # print('\n')
    # for MA1 in MatA:
    #     for MA2 in MatA:
    #         print(" err : ",npy.max(npy.abs(MA1-MA2)/npy.abs(MA1)))
    
    # plt.figure(1)
    # plt.subplot(2,2,1)
    # plt.plot(Kxj,'ro')
    # #plt.plot(Betaj,'bo')
    # plt.subplot(2,2,2)
    # plt.plot(1/Lbd1,'ro')
    # plt.plot(1/Lbd2,'bo')
    # plt.subplot(2,2,3)
    # plt.pcolormesh(Gam)
    # plt.colorbar()
    # plt.subplot(2,2,4)
    # plt.pcolormesh(matriceA)
    # plt.colorbar()
    # plt.show()
    # -----------------------
    # test lbd et gamma
    # -----------------------
    
    

    # propagatif
    # dk = 1/(10*MV.zf)
    # Kej = npy.linspace(0,npy.sqrt(wne2-wns2),10*freq) #npy.arange(wne,wns,dk)
    # Y = MV.fct_zeros_propag(freq,Kej)
    # print(Y)
    # Kxj = MV.calcul_valeurs_propres_propag(Kej,freq)
    # ModesProp = MV.modesPropagatifs(freq,Kxj,Z)
    # for ik in range(0,len(Kxj)):
    #     plt.figure(ik+1)
    #     plt.plot(Z,ModesProp[:,ik],'r+')
    # plt.show()
    # zeros = MV.calcul_valeurs_propres_propag(freq)
    # print(zeros[0:5])
    # plt.plot(Kej,Y.real,'r+')
    # plt.plot(zeros,0*zeros,'bo')
    # plt.show()             
    # radiatif
    # Gamma = npy.arange(wns2 - 4 , wns2 , 2 )
    # sortie = MV.modesRadiatifs(freq,Gamma,Z)
    # for kg in range(0,len(Gamma)):
    #     plt.figure(kg+1)
    #     plt.plot(Z,sortie[:,kg].real,'r+')
    #     plt.plot(Z,sortie[:,kg].imag,'b+')
    # plt.show()
