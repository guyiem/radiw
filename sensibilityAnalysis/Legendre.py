#coding: utf-8
import numpy as npy


# ------------------------------------
# definition of 1D-Legendre polynomials
# ------------------------------------

# renormalization constant
rnc1 = npy.sqrt(3)
rnc2 = npy.sqrt(5)
rnc3 = npy.sqrt(7)
# renormalization constant

class Legendre(object):

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
class MultivariateLegendre(object):

    def __init__(self,A=-npy.ones(6),B=npy.ones(6)):
        self.A = A
        self.B = B
        self.Legs = [ Legendre(a=A[k],b=B[k]) for k in range(0,6) ]
    
    def order1_mpolynomials(self):
        polys = []
        for k in range(0,6):
            ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
            ptmp[k] = self.Legs[k].leg1
            polys.append(ptmp)
        return polys

        
    def order2_mpolynomials(self):
        polys = []
        for k1 in range(0,6):
            for k2 in range(k1,6):
                if k1 == k2:
                    ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                    ptmp[k1] = self.Legs[k1].leg2
                    polys.append(ptmp)
                else:
                    ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                    ptmp[k1] = self.Legs[k1].leg1
                    ptmp[k2] = self.Legs[k2].leg1
                polys.append(ptmp)
        return polys


    def order3_mpolynomials(self):
        polys = []
        for k1 in range(0,6):
            ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
            ptmp[k1] = self.Legs[k1].leg3
            polys.append(ptmp)
        for k1 in range(0,6):
            for k2 in range(k1+1,6):
                ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                ptmp[k1] = self.Legs[k1].leg2
                ptmp[k2] = self.Legs[k2].leg1
                polys.append(ptmp)
                ptmp = [ self.Legs[k].leg0  for k in range(0,6) ]
                ptmp[k1] = self.Legs[k1].leg1
                ptmp[k2] = self.Legs[k2].leg2
                polys.append(ptmp)
        for k1 in range(0,6):
            for k2 in range(k1+1,6):
                for k3 in range(k2+1,6):
                    ptmp = [ self.Legs[k].leg0 for k in range(0,6) ]
                    ptmp[k1] = self.Legs[k1].leg1
                    ptmp[k2] = self.Legs[k2].leg1
                    ptmp[k3] = self.Legs[k3].leg1
                    polys.append(ptmp)
        return polys


    
def eval_mvl(mpoly,X):
    assert ( len(mpoly)==len(X)), ( " longueur inégale ")
    sortie = 1
    Xtmp = X
    for kp,poly in enumerate(mpoly):
        sortie *= poly(Xtmp[kp])
    return sortie


def regression(echants,rc,A=-npy.ones(6),B=npy.ones(6)):
    MVL = MultivariateLegendre(A,B)
    Polys0 =  [[ MVL.Legs[k].leg0 for k in range(0,6) ]]
    Polys1 = MVL.order1_mpolynomials()
    Polys2 = [] #MVL.order2_mpolynomials()
    Polys3 = [] #MVL.order3_mpolynomials()
    Polys = Polys3 + Polys2 + Polys1 + Polys0
    nbPolys = len(Polys0) + len(Polys1) + len(Polys2) + len(Polys3)
    nbEchants = len(echants)
    matA = npy.zeros((nbEchants,nbPolys))
    for ke,echant in enumerate(echants):
        for kmp,mpoly in enumerate(Polys):
            matA[ke,kmp] = eval_mvl(mpoly,echant)
    second_membre = npy.dot(matA.T,rc)
    matrice = npy.dot(matA.T,matA)
    print(npy.linalg.det(matrice))
    try:
        sortie = npy.linalg.solve(matrice,second_membre)
    except npy.linalg.LinAlgError:
        import ipdb ; ipdb.set_trace()        
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
        

                    
    
        
