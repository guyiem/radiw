#coding: utf-8

import numpy as npy


def supp_diag(M):
    """
    suppress diagonal of a square matrix.

    parameters : M, numpy array of shape (m,m)

    return : a numpy array of shape (m,m-1)
    """
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

    
def reductionAmplitudeMoitie(X,Y):
    """
    find point where amplitude of a curve ( X -> Y ) is divided by 2.
    
    parameters :
    X : ordonned numpy array of float of lenth m
    Y : numpy array of float of length m

    return :
    the float value corresponding to absciss point where amplitude is divided by two
    """

    if ((X[1:] - X[:-1])<=0).any():
        raise ValueError( " X is not an ordonned array ")
        
    Xtmp = X -  X[0]
    if X.shape != Y.shape:
        print(X.shape,Y.shape)
        raise ValueError("X and Y doesn't have same shape ")
    #if X[0] != 0:
    #    print('X ne vaut pas zéro à son premier point')
    imax = npy.argmax(Y)
    if imax != 0:
        raise ValueError('Y maximum is not at the first point')
    ymax = Y[imax]
    compteur = 0
    while (Y[compteur]>ymax/2) & (compteur<=Xtmp.size):
        compteur +=1
    y0 = Y[compteur-1]
    y1 = Y[compteur]
    x0 = Xtmp[compteur-1]
    x1 = Xtmp[compteur]
    if compteur>Xtmp.size:
        raise IndexError( " out of range : amplitude division by two is out of X ") 
    # i05 = npy.argmin((Y-ymax/2)**2)
    # if Y[i05] == ymax/2:
    #     #print('dessus!')
    #     return X[i05]
    # elif Y[i05] > ymax/2:
    #     #print('indice avant!')
    #     x0 = X[i05]
    #     x1 = X[i05+1]
    #     y0 = Y[i05]
    #     y1 = Y[i05+1]
    # elif Y[i05] < ymax/2:
    #     #print('indice après!')
    #     x0 = X[i05-1]
    #     x1 = X[i05]
    #     y0 = Y[i05-1]
    #     y1 = Y[i05]
    # #print(x0,x1,y0,y1)
    toto = (ymax/2 - y0)*(x1-x0)/(y1-y0) + x0 + X[0]
    return toto
