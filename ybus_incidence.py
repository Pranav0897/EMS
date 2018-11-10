# Generated with SMOP  0.41-beta
from . import *
import numpy as np # from numpy import *
# ybus_incidence.m

    ## Formulation of Ybus by singular transformation method (With Transformer Tap settings and Shunt Admittances)
    
#@function
def ybus_incidence(r=None,x=None,b=None,*args,**kwargs):
    # varargin = ybus_incidence.varargin
    # nargin = ybus_incidence.nargin

    global fb,tb,nbranch,nbus,linedatas,baseMVA,busdatas
    #linedatas(:,6) = 1;
    # tap=linedatas(np.arange(),6)
    tap=linedatas[:,5]
# ybus_incidence.m:5
    
    # GS=busdatas(np.arange(),11)
    GS=busdatas[:,10]
# ybus_incidence.m:6
    
    # BS=busdatas(np.arange(),12)
    BS=busdatas[:,11]
# ybus_incidence.m:7
    
    Ysh=(GS + np.dot(1j,BS)) / baseMVA
# ybus_incidence.m:8
    
    Z=r + np.dot(1j,x)
# ybus_incidence.m:9
    
    Y=1.0 / Z
# ybus_incidence.m:10
    ## Formation of Bus Incidence matrix A (signs: comes in is -1, goes out is +1)
    A=zeros(nbranch + nbus,nbus)
# ybus_incidence.m:12
    for i in np.arange(1,nbus).reshape(-1):
        for j in np.arange(1,nbus).reshape(-1):
            if (i == j):
                A[i,i]=1
# ybus_incidence.m:16
    
    for i in np.arange(nbus + 1,nbus + nbranch).reshape(-1):
        A[i,fb(i - nbus)]=1
# ybus_incidence.m:21
        A[i,tb(i - nbus)]=- 1
# ybus_incidence.m:22
    
    ## Calculation of primitive matrix
    Yprimitive=zeros(nbranch + nbus,1)
# ybus_incidence.m:25
    # For buses:
    for i in np.arange(1,nbranch).reshape(-1):
        Yprimitive[fb(i)]=Yprimitive(fb(i)) + np.dot(1j,b(i)) / 2 + np.dot((1 - tap(i)),Y(i)) / tap(i) ** 2
# ybus_incidence.m:28
        Yprimitive[tb(i)]=Yprimitive(tb(i)) + np.dot(1j,b(i)) / 2 + np.dot((tap(i) - 1),Y(i)) / tap(i)
# ybus_incidence.m:29
    
    Yprimitive[np.arange(1,nbus)]=Yprimitive(np.arange(1,nbus)) + Ysh
# ybus_incidence.m:31
    
    # Branches:
    for i in np.arange(1,nbranch).reshape(-1):
        Yprimitive[i + nbus]=Y(i) / tap(i)
# ybus_incidence.m:34
    
    ## Bus Admittance matrix:
    Ybus=np.dot(np.dot(A.T,diag(Yprimitive)),A)
# ybus_incidence.m:37
    