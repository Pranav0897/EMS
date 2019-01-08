# Generated with SMOP  0.41-beta
from . import *
import numpy as np 
# power_inj.m

## Calculation of Power Injections (p.u)
# V - Voltage Magnitude pu
# phi - Voltage Angle in radians
    
#@function
def power_inj(V=None,phi=None,Ybus=None,*args,**kwargs):
    # varargin = power_inj.varargin
    # nargin = power_inj.nargin

    # function Si = power_inj(V,phi,Ybus)
    global nbus,bus_type,Pgen,Qgen,Qload,Pload
    G=np.real(Ybus)
# power_inj.m:7
    
    B=np.imag(Ybus)
    ## power injection calculations From WLS
    Pi=np.zeros(shape=(nbus,1))
    Qi=np.zeros(shape=(nbus,1))
    # Pi_a = zeros(nbus,1);
    # Qi_a = zeros(nbus,1);
    for i in np.arange(1,nbus).reshape(-1):
        for j in np.arange(1,nbus).reshape(-1):
            Pi[i]=Pi(i) + np.dot(np.dot(V(i),V(j)),(np.dot(G(i,j),cos(phi(i) - phi(j))) + np.dot(B(i,j),sin(phi(i) - phi(j)))))
            Qi[i]=Qi(i) + np.dot(np.dot(V(i),V(j)),(np.dot(G(i,j),sin(phi(i) - phi(j))) - np.dot(B(i,j),cos(phi(i) - phi(j)))))
            # Pi_a(i) = Pi(i) + V_a(i)*V_a(j)*(G(i,j)*cos(phi_a(i)-phi_a(j)) +B(i,j)*sin(phi_a(i)-phi_a(j)));
            # Qi_a(i) = Qi(i) + V_a(i)*V_a(j)*(G(i,j)*sin(phi_a(i)-phi_a(j)) -B(i,j)*cos(phi_a(i)-phi_a(j)));
    
    # for i = 1:nbus
#     if (bus_type(i)==1 || bus_type(i) ==2)
#         Pi(i)=Pi_a(i);
#         Qi(i)=Qi_a(i);
#     end
# end
# for i = 1:nbus
#     if Pgen(i)==0 && Qgen(i) ==0 && Pload(i)==0 && Qload(i)==0
#        Pi(i)=Pi_a(i);
#        Qi(i)=Qi_a(i); 
#     end
# end