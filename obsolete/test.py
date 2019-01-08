# Generated with SMOP  0.41-beta
from . import *
import numpy as np # from numpy import *
import copy
from measurement14 import *
# test.m

# clc
# clear('all')
global fb,tb,nbranch,nbus,linedatas,busdatas,V_nr,phi_nr,test_case,r_perturbed,x_perturbed,b_perturbed,r_nominal,x_nominal,b_nominal,baseMVA,Phi_true,V_true,V_act,phi_act,Pgen,Qgen,Pload,Qload,bus_type
#[changeV,changeDel] = attackfn1();
#changeV=0;
#changeDel=0;
## Initials:
test_case=14
MC_tests=10
display_SE_results=1
Ps=0
Pm=0
## Variance of Voltage magnitude, Active and reactive Power Injections and flows:
sigma_v=0.01
sigma_phi=0.01
sigma_pi=0.01
sigma_qi=0.01
sigma_pf=0.01
sigma_qf=0.01
rho=0.0

## Reading Nominal Line Parameters and Ybus arrangement:
linedatas=linedata.linedata()

fb=linedatas(np.arange(),1)


tb=linedatas(np.arange(),2)
nbranch=len(fb)


nbus=copy.deepcopy(test_case)

negatives=0

r_nominal=linedatas(np.arange(),3)

x_nominal=linedatas(np.arange(),4)

b_nominal=linedatas(np.arange(),5)

busdatas=busdata()


bus_type=busdatas(np.arange(),2)

Pgen=busdatas(np.arange(),5) / baseMVA

Qgen=busdatas(np.arange(),6) / baseMVA

Pload=busdatas(np.arange(),7) / baseMVA

Qload=busdatas(np.arange(),8) / baseMVA

Ybus_nominal,A_incidence=ybus_incidence(r_nominal,x_nominal,b_nominal,nargout=2)

G=real(Ybus_nominal)


B=imag(Ybus_nominal)

V_error_seq=[]

phi_error_seq=[]

V_SE_seq=[]

phi_SE_seq=[]

V_nr_seq=[]

phi_nr_seq=[]

sigma_x_v_seq=[]

sigma_x_phi_seq=[]


V_act,phi_act=newton(Ybus_nominal,nargout=2)

V_nr=copy.deepcopy(V_act)

phi_nr=copy.deepcopy(phi_act)


for MC_test in np.arange(1,MC_tests).reshape(-1):
    # V_nr = np.random.normal(V_nr, 0.001 .* V_nr);
    #  phi_nr = np.random.normal(phi_nr, 0.02 .* phi_nr);
    phi_nr_dg=np.dot(180 / np.pi,phi_nr)
    ## WLS State Estimation
    # Shunt Admittance Matrix Formation: # Off-diagonals are the mutual admittances between the respective nodes
    bbus=np.zeros((nbus,nbus))

    for k in np.arange(1,nbranch).reshape(-1):
        bbus[fb(k),tb(k)]=b_nominal(k) / 2
        bbus[tb(k),fb(k)]=bbus(fb(k),tb(k))
    meas_calc
    wls
    phi_SE_dg=np.dot(180 / np.pi,phi_SE)
    ## Collecting V and Phi errors in each iterarion
    V_error=V_nr - V_SE
    phi_error=phi_nr_dg - phi_SE_dg
    ## Display the WLS results
    if display_SE_results:
        disp('_________________State Estimation_________________')
        disp('Bus V_act V_SE V_NR V_Er Ph_act Ph_SE Ph_NR Ph_Er')
        for m in np.arange(1,nbus).reshape(-1):
            fprintf('%3g',m)
            fprintf('%8.3f',V_act(m))
            fprintf('%8.3f',V_SE(m))
            fprintf('%7.3f',V_nr(m))
            fprintf('%21.16f',V_nr(m) - V_SE(m))
            fprintf(' %8.3f',np.dot(phi_act(m),180) / pi)
            fprintf(' %8.3f',phi_SE_dg(m))
            fprintf(' %8.3f',phi_nr_dg(m))
            fprintf('%21.16f',phi_nr_dg(m) - phi_SE_dg(m))
            fprintf('\\n')
        disp('__________________________________________________')
    fprintf('MC Trial Number: ')
    fprintf('%g',MC_test)
    fprintf('\\n')
    if iters > 1:
        fprintf('WLS Iterations = %4d',iters)
        fprintf('\\n')
    #end # of display each 10 trial

