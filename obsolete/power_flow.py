# Generated with SMOP  0.41-beta
from . import *
import numpy as np # from numpy import *
# power_flow.m

## Calculation of Line Power flows (p.u)
# V - Voltage Magnitude pu
# phi - Voltage Angle in radians
    
#@function
def power_flow(V=None,phi=None,Ybus=None,*args,**kwargs):
#     varargin = power_flow.varargin
#     nargin = power_flow.nargin

    global fb,tb,nbus,nbranch,b_nominal
    G=np.real(Ybus)
    B=np.imag(Ybus)
    ## Polar coordination:
    # Shunt Admittance Matrix Formation: # Off-diagonals are the mutual admittances between the respective nodes
    bbus_perturbed=np.zeros((nbus,nbus))
    for k in np.arange(1,nbranch).reshape(-1):
        bbus_perturbed[fb(k),tb(k)]=b_nominal(k) / 2
        bbus_perturbed[tb(k),fb(k)]=bbus_perturbed(fb(k),tb(k))
    # power flows calculation
    for i in np.arange(1,nbranch).reshape(-1):
        m=fb(i)
        n=tb(i)
        Pij[i]=np.dot(- V(m) ** 2,(G(m,n))) + np.dot(np.dot(V(m),V(n)),(np.dot(G(m,n),cos(phi(m) - phi(n))) + np.dot(B(m,n),sin(phi(m) - phi(n)))))
        Qij[i]=np.dot(V(m) ** 2,(B(m,n) - bbus_perturbed(m,n))) + np.dot(np.dot(V(m),V(n)),(np.dot(G(m,n),sin(phi(m) - phi(n))) - np.dot(B(m,n),cos(phi(m) - phi(n)))))
        Pji[i]=np.dot(- V(n) ** 2,(G(n,m))) + np.dot(np.dot(V(n),V(m)),(np.dot(G(n,m),cos(phi(n) - phi(m))) + np.dot(B(n,m),sin(phi(n) - phi(m)))))
        Qji[i]=np.dot(V(n) ** 2,(B(n,m) - bbus_perturbed(n,m))) + np.dot(np.dot(V(n),V(m)),(np.dot(G(n,m),sin(phi(n) - phi(m))) - np.dot(B(n,m),cos(phi(n) - phi(m)))))

    