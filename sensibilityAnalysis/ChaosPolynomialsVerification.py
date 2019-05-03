#coding: utf-8
import numpy as npy

from Legendre import *

def orthonormality_check():
    a = 5
    b = 10
    Y = npy.linspace(a,b,1001)
    dy = Y[1] - Y[0]
    L1 = leg1(Y,a=a,b=b)
    L2 = leg2(Y,a=a,b=b)
    L3 = leg3(Y,a=a,b=b)
    print(" dy : ",dy)
    print(" normes :")
    print(npy.sum( (L1*L1)[1:]/(b-a))*dy,npy.sum( (L2*L2)[1:]/(b-a))*dy,npy.sum( (L3*L3)[1:]/(b-a))*dy)
    print(" PS : ")
    print(npy.sum( (L1*L2)[1:]/(b-a))*dy,npy.sum( (L1*L3)[1:]/(b-a))*dy,npy.sum( (L2*L3)[1:]/(b-a))*dy)    

