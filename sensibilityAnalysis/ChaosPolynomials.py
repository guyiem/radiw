#coding: utf-8
import numpy as npy


class MultiPoly(object):

    def __init__(self,mpoly,degrees):
        self.mpoly = mpoly
        self.degrees = degrees

    def degree(self):
        return npy.sum(self.degrees)
        
    def eval_mvp(self,X):
        assert ( len(self.mpoly)==len(X)), ( " longueur inégale ")
        sortie = 1
        for kp,poly in enumerate(self.mpoly):
            sortie *= poly(X[kp])
        return sortie


# ------------------------------------
# definition of 1D-Legendre polynomials
# ------------------------------------
# renormalization constant
rnc1 = npy.sqrt(3)
rnc2 = npy.sqrt(5)
rnc3 = npy.sqrt(7)
# renormalization constant

class Legendre3(object):

    def __init__(self,a=-1,b=1):
        self.a = a
        self.b = b
        

    def leg0(self,y):
        assert ( npy.min(y) >= self.a), ( " y < a ")
        assert ( npy.max(y) <= self.b), ( " y > b ")
        return 1

    def leg1(self,y):
        assert ( npy.min(y) >= self.a), ( " y < a ")
        assert ( npy.max(y) <= self.b), ( " y > b ")
        return rnc1 * ( 2/(self.b-self.a)*( y - self.a ) - 1 )

    def leg2(self,y):
        assert ( npy.min(y) >= self.a), ( " y < a ")
        assert ( npy.max(y) <= self.b), ( " y > b ")
        return rnc2 * 0.5 * ( 3 * ( 2/(self.b-self.a)*(y-self.a) - 1 )**2 - 1 )

    def leg3(self,y):
        assert ( npy.min(y) >= self.a), ( " y < a ")
        assert ( npy.max(y) <= self.b), ( " y > b ")
        x = 2/(self.b-self.a)*( y - self.a) - 1
        return rnc3 * 0.5 * ( 5*x**3 - 3*x )

    
# -----------------------------------------
# 6D polynomials up to order 3
# -----------------------------------------
class MultivariateLegendre3(object):

    def __init__(self,A=-npy.ones(6),B=npy.ones(6)):
        self.A = A
        self.B = B
        self.Legs = [ Legendre3(a=A[k],b=B[k]) for k in range(0,6) ]
        # Polys = []
        # Polys += self.order0_mpolynomials()
        # Polys += self.order1_mpolynomials()
        # Polys += self.order2_mpolynomials()
        # Polys += self.order3_mpolynomials()
        # self.Polys = Polys
              
    def order0_mpolynomials(self):
        mpoly = [ self.Legs[k].leg0 for k in range(0,6) ]
        degrees = npy.array([0,0,0,0,0,0])
        return [MultiPoly(mpoly,degrees)]
        
    def order1_mpolynomials(self):
        polys = []
        for k in range(0,6):
            degrees = npy.array([0,0,0,0,0,0])
            ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
            ptmp[k] = self.Legs[k].leg1
            degrees[k]  = 1
            polys.append(MultiPoly(ptmp,degrees))
        return polys
        
    def order2_mpolynomials(self):
        polys = []
        for k1 in range(0,6):
            for k2 in range(k1,6):
                if k1 == k2:
                    degrees = npy.array([0,0,0,0,0,0])
                    ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                    ptmp[k1] = self.Legs[k1].leg2
                    degrees[k1] = 2
                else:
                    degrees = npy.array([0,0,0,0,0,0])
                    ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                    ptmp[k1] = self.Legs[k1].leg1
                    ptmp[k2] = self.Legs[k2].leg1
                    degrees[k1] = 1
                    degrees[k2] = 1
                polys.append(MultiPoly(ptmp,degrees))
        return polys

    def order3_mpolynomials(self):
        polys = []
        for k1 in range(0,6):
            degrees = npy.array([0,0,0,0,0,0])            
            ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
            ptmp[k1] = self.Legs[k1].leg3
            degrees[k1] = 3
            polys.append(MultiPoly(ptmp,degrees))
        for k1 in range(0,6):
            for k2 in range(k1+1,6):
                degrees = npy.array([0,0,0,0,0,0])
                ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                ptmp[k1] = self.Legs[k1].leg2
                ptmp[k2] = self.Legs[k2].leg1
                degrees[k1] = 2
                degrees[k2] = 1
                polys.append(MultiPoly(ptmp,degrees))
                degrees = npy.array([0,0,0,0,0,0])
                ptmp = [ self.Legs[k].leg0  for k in range(0,6) ]
                ptmp[k1] = self.Legs[k1].leg1
                ptmp[k2] = self.Legs[k2].leg2
                degrees[k1] = 1
                degrees[k2] = 2
                polys.append(MultiPoly(ptmp,degrees))
        for k1 in range(0,6):
            for k2 in range(k1+1,6):
                for k3 in range(k2+1,6):
                    degrees = npy.array([0,0,0,0,0,0])
                    ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                    ptmp[k1] = self.Legs[k1].leg1
                    ptmp[k2] = self.Legs[k2].leg1
                    ptmp[k3] = self.Legs[k3].leg1
                    degrees[k1] = 1
                    degrees[k2] = 1
                    degrees[k3] = 1
                    polys.append(MultiPoly(ptmp,degrees))
        return polys

    # def regression(self,Xsamples,Ysamples,regularization=0.0):
    #     nbPolys = len(self.Polys)
    #     nbSamples = len(Xsamples)
    #     matA = npy.zeros((nbSamples,nbPolys))
    #     for ke,echant in enumerate(Xsamples):
    #         for kmp,mpoly in enumerate(self.Polys):
    #             matA[ke,kmp] = mpoly.eval_mvp(echant)
    #     second_membre = npy.dot(matA.T,Ysamples)
    #     matrice = npy.dot(matA.T,matA)
    #     matrice +=  + regularization*npy.eye(matrice.shape[0])
    #     print(npy.linalg.det(matrice))
    #     try:
    #         sortie = npy.linalg.solve(matrice,second_membre)
    #     except npy.linalg.LinAlgError:
    #         import ipdb ; ipdb.set_trace()        
    #     return sortie


    
class ChaosPolynomialsModel3(object):

    def __init__(self,A,B,Xsamples,Ysamples):
        self.A = A
        self.B = B
        MVL = MultivariateLegendre3(A=A,B=B)
        Polys = []
        Polys += MVL.order0_mpolynomials()
        Polys += MVL.order1_mpolynomials()
        Polys += MVL.order2_mpolynomials()
        Polys += MVL.order3_mpolynomials()
        self.Polys = Polys
        self.Poids = self.regression(Xsamples,Ysamples,regularization=0.0)
        assert ( len(self.Poids) == len(self.Polys) ), ( " pb ! ")

        
    def regression(self,Xsamples,Ysamples,regularization=0.0):
        nbPolys = len(self.Polys)
        nbSamples = len(Xsamples)
        matA = npy.zeros((nbSamples,nbPolys))
        for ke,echant in enumerate(Xsamples):
            for kmp,mpoly in enumerate(self.Polys):
                matA[ke,kmp] = mpoly.eval_mvp(echant)
        second_membre = npy.dot(matA.T,Ysamples)
        matrice = npy.dot(matA.T,matA)
        matrice +=  + regularization*npy.eye(matrice.shape[0])
        print(npy.linalg.det(matrice))
        try:
            sortie = npy.linalg.solve(matrice,second_membre)
        except npy.linalg.LinAlgError:
            import ipdb ; ipdb.set_trace()        
        return sortie

    def eval_model(self,sample):
        sortie = 0
        for kp,poids in enumerate(self.Poids):
            sortie += poids*self.Polys[kp].eval_mvp(sample)
        return sortie
        

# def order1_polynomials(a=-1,b=1):
#     polys = []
#     for k in range(0,6):
#         ptmp = [ 0 for k in range(0,6) ]
#         ptmp[k] = 1
#         polys.append(ptmp)
#     return polys

        
# def order2_polynomials(a=-1,b=1):
#     polys = []
#     for k1 in range(0,6):
#         for k2 in range(k1,6):
#             if k1 == k2:
#                 ptmp = [ 0 for k in range(0,6) ]
#                 ptmp[k1] = 2
#                 polys.append(ptmp)
#             else:
#                 ptmp = [ 0 for k in range(0,6) ]
#                 ptmp[k1] = 1
#                 ptmp[k2] = 1
#                 polys.append(ptmp)
#     return polys


# def order3_polynomials(a=-1,b=1):
#     polys = []
#     for k1 in range(0,6):
#         ptmp = [ 0 for k in range(0,6) ]
#         ptmp[k1] = 3
#         polys.append(ptmp)
#     for k1 in range(0,6):
#         for k2 in range(k1+1,6):
#             ptmp = [ 0 for k in range(0,6) ]
#             ptmp[k1] = 2
#             ptmp[k2] = 1
#             polys.append(ptmp)
#             ptmp = [ 0 for k in range(0,6) ]
#             ptmp[k1] = 1
#             ptmp[k2] = 2
#             polys.append(ptmp)
#     for k1 in range(0,6):
#         for k2 in range(k1+1,6):
#             for k3 in range(k2+1,6):
#                 ptmp = [ 0 for k in range(0,6) ]
#                 ptmp[k1] = 1
#                 ptmp[k2] = 1
#                 ptmp[k3] = 1
#                 polys.append(ptmp)
#     return polys
        

                    
    
        
