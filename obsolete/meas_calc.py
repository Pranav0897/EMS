# Generated with SMOP  0.41-beta
from . import *
import numpy as np # from numpy import *
# meas_calc.m

## Determining the Number of measurements:
def get_num_meas():
    n_v_meas=len(v_meas_bus_nr)

    n_pi_meas=len(pi_meas_bus_nr)

    n_qi_meas=len(qi_meas_bus_nr)

    n_pf_meas=len(pf_meas_bus_nr)

    n_qf_meas=len(qf_meas_bus_nr)

    n_phi_meas=len(pmu_meas_bus_nr)

    ## Variance Covariance Matrix(weighting Matrix) construction:
    sigma_vector=np.concat([np.dot(sigma_v,np.ones((n_v_meas,1))),np.dot(sigma_pi,np.ones((n_pi_meas,1))),np.dot(sigma_qi,np.ones((n_qi_meas,1))),np.dot(sigma_pf,np.ones((n_pf_meas,1))),np.dot(sigma_qf,np.ones((n_qf_meas,1) )),np.dot(sigma_phi,np.ones((n_phi_meas,1)))])
    sigma_square=np.diag(sigma_vector) ** 2 # assuming sigma_vector is a 1D array

    ## Voltage Measurements:
    v_meas=V_nr(v_meas_bus_nr)

    ## PMU Measurements:
    if pmu_meas_bus_nr:
        phi_meas=phi_nr(pmu_meas_bus_nr(np.arange(),1),1) - phi_nr(pmu_meas_bus_nr(np.arange(),2),1)
    else:
        phi_meas=[]

    ## Power Injection Measurements:
    Pi_nr,Qi_nr=power_inj(V_nr,phi_nr,Ybus_nominal,nargout=2)

    pi_meas=Pi_nr(pi_meas_bus_nr)

    qi_meas=Qi_nr(qi_meas_bus_nr)

    ## Power Flow Measuremets:
    Pij,Qij,Pji,Qji=power_flow(V_nr,phi_nr,Ybus_nominal,nargout=4)

    # Active power flow:
    pf_meas=[]
    for i in np.arange(1,len(pf_meas_bus_nr)).reshape(-1):
        m=pf_meas_bus_nr(i,1)
        n=pf_meas_bus_nr(i,2)
        for p in np.arange(1,nbranch).reshape(-1):
            if m == fb(p) and n == tb(p):
                pf_meas[i,1]=Pij(p)
            elif m == tb(p) and n == fb(p):
                pf_meas[i,1]=Pji(p)

    # ReActive power flow:
    qf_meas=[]
    # meas_calc.m:46
    for i in np.arange(1,len(qf_meas_bus_nr)).reshape(-1):
        m=qf_meas_bus_nr(i,1)
    # meas_calc.m:48
        n=qf_meas_bus_nr(i,2)
    # meas_calc.m:49
        for p in np.arange(1,nbranch).reshape(-1):
            if m == fb(p) and n == tb(p):
                qf_meas[i,1]=Qij(p)
    # meas_calc.m:52
            elif m == tb(p) and n == fb(p):
                qf_meas[i,1]=Qji(p)
    # meas_calc.m:54

    ## Measurement Perturbation:
    v_meas_perturbed=np.random.normal(v_meas,np.dot(sigma_v,v_meas))
    # meas_calc.m:59
    pi_meas_perturbed=np.random.normal(pi_meas,np.dot(sigma_pi,abs(pi_meas)))
    # meas_calc.m:60
    qi_meas_perturbed=np.random.normal(qi_meas,np.dot(sigma_qi,abs(qi_meas)))
    # meas_calc.m:61
    pf_meas_perturbed=np.random.normal(pf_meas,np.dot(sigma_pf,abs(pf_meas)))
    # meas_calc.m:62
    qf_meas_perturbed=np.random.normal(qf_meas,np.dot(sigma_qf,abs(qf_meas)))
    # meas_calc.m:63
    phi_meas_perturbed=np.random.normal(phi_meas,np.dot(sigma_phi,abs(phi_meas)))
    # meas_calc.m:64
    ## Perturbed Measurement Vector composition:
    y_perturbed=np.concat([v_meas_perturbed,pi_meas_perturbed,qi_meas_perturbed,pf_meas_perturbed,qf_meas_perturbed,phi_meas_perturbed])
    # meas_calc.m:66