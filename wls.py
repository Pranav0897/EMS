# Generated with SMOP  0.41-beta
from . import *
import numpy as np # from numpy import *
# wls.m

max_iters=50

tol=1e-06

## State Vector initialization:
V_SE=np.ones(shape=(nbus,1))

V_SE[v_meas_bus_nr,1]=v_meas_perturbed

phi_SE=np.zeros(shape=(nbus,1))

# if pmu_meas_bus_nr
# phi_SE(pmu_meas_bus_nr(:,2),1) = - phi_meas_perturbed; # put the phases az inits
# end
state=np.concat([[phi_SE(np.arange(2,end()))],[V_SE]])

iters=0
# wls.m:11
converged=0
# wls.m:12
while (np.logical_not(converged) and iters < max_iters):

    iters=iters + 1
# wls.m:14
    f1=V_SE(v_meas_bus_nr,1)
# wls.m:16
    f2=zeros(n_pi_meas,1)
# wls.m:17
    f3=zeros(n_qi_meas,1)
# wls.m:18
    f4=zeros(n_pf_meas,1)
# wls.m:19
    f5=zeros(n_qf_meas,1)
# wls.m:20
    if pmu_meas_bus_nr:
        f6=phi_SE(pmu_meas_bus_nr(np.arange(),1),1) - phi_SE(pmu_meas_bus_nr(np.arange(),2),1)
# wls.m:22
    else:
        f6=[]
# wls.m:23
    # V_SE = V_nr;
# phi_SE = phi_nr;
#Real power injection calculation OK
    for i in np.arange(1,n_pi_meas).reshape(-1):
        m=pi_meas_bus_nr(i)
# wls.m:29
        for k in np.arange(1,nbus).reshape(-1):
            f2[i]=f2(i) + np.dot(np.dot(V_SE(m),V_SE(k)),(np.dot(G(m,k),cos(phi_SE(m) - phi_SE(k))) + np.dot(B(m,k),sin(phi_SE(m) - phi_SE(k)))))
# wls.m:31
    # Reactive power injection calculations OK
    for i in np.arange(1,n_qi_meas).reshape(-1):
        m=qi_meas_bus_nr(i)
# wls.m:36
        for k in np.arange(1,nbus).reshape(-1):
            f3[i]=f3(i) + np.dot(np.dot(V_SE(m),V_SE(k)),(np.dot(G(m,k),sin(phi_SE(m) - phi_SE(k))) - np.dot(B(m,k),cos(phi_SE(m) - phi_SE(k)))))
# wls.m:38
    # Real power flows calculation OK
    for i in np.arange(1,n_pf_meas).reshape(-1):
        m=pf_meas_bus_nr(i,1)
# wls.m:43
        n=pf_meas_bus_nr(i,2)
# wls.m:44
        f4[i]=np.dot(- V_SE(m) ** 2,(G(m,n))) + np.dot(np.dot(V_SE(m),V_SE(n)),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n)))))
# wls.m:45
    # Reactive power flows calculation OK
    for i in np.arange(1,n_qf_meas).reshape(-1):
        m=qf_meas_bus_nr(i,1)
# wls.m:49
        n=qf_meas_bus_nr(i,2)
# wls.m:50
        f5[i]=np.dot(V_SE(m) ** 2,(B(m,n) - bbus(m,n))) + np.dot(np.dot(V_SE(m),V_SE(n)),(np.dot(G(m,n),sin(phi_SE(m) - phi_SE(n))) - np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n)))))
# wls.m:51
    f=np.concat([[f1],[f2],[f3],[f4],[f5],[f6]])
# wls.m:53
    # g11 - Derivative of V_SE with respect to angles: All Zeros
    g11=zeros(n_v_meas,nbus - 1)
# wls.m:56
    g12=zeros(n_v_meas,nbus)
# wls.m:58
    for k in np.arange(1,n_v_meas).reshape(-1):
        for n in np.arange(1,nbus).reshape(-1):
            if n == k:
                g12[k,n]=1
# wls.m:62
    # g21 - Derivative of Real Power Injections with Angles
    g21=zeros(n_pi_meas,nbus - 1)
# wls.m:67
    for i in np.arange(1,n_pi_meas).reshape(-1):
        m=pi_meas_bus_nr(i)
# wls.m:69
        for k in np.arange(1,(nbus - 1)).reshape(-1):
            if k + 1 == m:
                for n in np.arange(1,nbus).reshape(-1):
                    g21[i,k]=g21(i,k) + np.dot(np.dot(V_SE(m),V_SE(n)),(np.dot(- G(m,n),sin(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n)))))
# wls.m:73
                g21[i,k]=g21(i,k) - np.dot(V_SE(m) ** 2,B(m,m))
# wls.m:75
            else:
                g21[i,k]=np.dot(np.dot(V_SE(m),V_SE(k + 1)),(np.dot(G(m,k + 1),sin(phi_SE(m) - phi_SE(k + 1))) - np.dot(B(m,k + 1),cos(phi_SE(m) - phi_SE(k + 1)))))
# wls.m:77
    # g22 - Derivative of Real Power Injections with V_SE
    g22=zeros(n_pi_meas,nbus)
# wls.m:82
    for i in np.arange(1,n_pi_meas).reshape(-1):
        m=pi_meas_bus_nr(i)
# wls.m:84
        for k in np.arange(1,(nbus)).reshape(-1):
            if k == m:
                for n in np.arange(1,nbus).reshape(-1):
                    g22[i,k]=g22(i,k) + np.dot(V_SE(n),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n)))))
# wls.m:88
                g22[i,k]=g22(i,k) + np.dot(V_SE(m),G(m,m))
# wls.m:90
            else:
                g22[i,k]=np.dot(V_SE(m),(np.dot(G(m,k),cos(phi_SE(m) - phi_SE(k))) + np.dot(B(m,k),sin(phi_SE(m) - phi_SE(k)))))
# wls.m:92
    # g31 - Derivative of Reactive Power Injections with Angles
    g31=zeros(n_qi_meas,nbus - 1)
# wls.m:97
    for i in np.arange(1,n_qi_meas).reshape(-1):
        m=qi_meas_bus_nr(i)
# wls.m:99
        for k in np.arange(1,(nbus - 1)).reshape(-1):
            if k + 1 == m:
                for n in np.arange(1,nbus).reshape(-1):
                    g31[i,k]=g31(i,k) + np.dot(np.dot(V_SE(m),V_SE(n)),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n)))))
# wls.m:103
                g31[i,k]=g31(i,k) - np.dot(V_SE(m) ** 2,G(m,m))
# wls.m:105
            else:
                g31[i,k]=np.dot(np.dot(V_SE(m),V_SE(k + 1)),(np.dot(- G(m,k + 1),cos(phi_SE(m) - phi_SE(k + 1))) - np.dot(B(m,k + 1),sin(phi_SE(m) - phi_SE(k + 1)))))
# wls.m:107
    # g32 - Derivative of Reactive Power Injections with V_SE
    g32=zeros(n_qi_meas,nbus)
# wls.m:112
    for i in np.arange(1,n_qi_meas).reshape(-1):
        m=qi_meas_bus_nr(i)
# wls.m:114
        for k in np.arange(1,(nbus)).reshape(-1):
            if k == m:
                for n in np.arange(1,nbus).reshape(-1):
                    g32[i,k]=g32(i,k) + np.dot(V_SE(n),(np.dot(G(m,n),sin(phi_SE(m) - phi_SE(n))) - np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n)))))
# wls.m:118
                g32[i,k]=g32(i,k) - np.dot(V_SE(m),B(m,m))
# wls.m:120
            else:
                g32[i,k]=np.dot(V_SE(m),(np.dot(G(m,k),sin(phi_SE(m) - phi_SE(k))) - np.dot(B(m,k),cos(phi_SE(m) - phi_SE(k)))))
# wls.m:122
    # g41 - Derivative of Real Power Flows with Angles
    g41=zeros(n_pf_meas,nbus - 1)
# wls.m:127
    for i in np.arange(1,n_pf_meas).reshape(-1):
        m=pf_meas_bus_nr(i,1)
# wls.m:129
        n=pf_meas_bus_nr(i,2)
# wls.m:130
        for k in np.arange(1,(nbus - 1)).reshape(-1):
            if k + 1 == m:
                g41[i,k]=np.dot(np.dot(- V_SE(m),V_SE(n)),(np.dot(G(m,n),sin(phi_SE(m) - phi_SE(n))) - np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n)))))
# wls.m:133
            else:
                if k + 1 == n:
                    g41[i,k]=np.dot(np.dot(V_SE(m),V_SE(n)),(np.dot(G(m,n),sin(phi_SE(m) - phi_SE(n))) - np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n)))))
# wls.m:135
                else:
                    g41[i,k]=0
# wls.m:137
    # g42 - Derivative of Real Power Flows with V_SE
    g42=zeros(n_pf_meas,nbus)
# wls.m:143
    for i in np.arange(1,n_pf_meas).reshape(-1):
        m=pf_meas_bus_nr(i,1)
# wls.m:145
        n=pf_meas_bus_nr(i,2)
# wls.m:146
        for k in np.arange(1,nbus).reshape(-1):
            if k == m:
                g42[i,k]=np.dot(V_SE(n),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n))))) - np.dot(np.dot(2,G(m,n)),V_SE(m))
# wls.m:149
            else:
                if k == n:
                    g42[i,k]=np.dot(V_SE(m),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n)))))
# wls.m:151
                else:
                    g42[i,k]=0
# wls.m:153
    # g51 - Derivative of Reactive Power Flows with Angles
    g51=zeros(n_qf_meas,nbus - 1)
# wls.m:159
    for i in np.arange(1,n_qf_meas).reshape(-1):
        m=qf_meas_bus_nr(i,1)
# wls.m:161
        n=qf_meas_bus_nr(i,2)
# wls.m:162
        for k in np.arange(1,(nbus - 1)).reshape(-1):
            if k + 1 == m:
                g51[i,k]=np.dot(np.dot(V_SE(m),V_SE(n)),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n)))))
# wls.m:165
            else:
                if k + 1 == n:
                    g51[i,k]=np.dot(np.dot(- V_SE(m),V_SE(n)),(np.dot(G(m,n),cos(phi_SE(m) - phi_SE(n))) + np.dot(B(m,n),sin(phi_SE(m) - phi_SE(n)))))
# wls.m:167
                else:
                    g51[i,k]=0
# wls.m:169
    # g52 - Derivative of Reactive Power Flows with V_SE
    g52=zeros(n_qf_meas,nbus)
# wls.m:175
    for i in np.arange(1,n_qf_meas).reshape(-1):
        m=qf_meas_bus_nr(i,1)
# wls.m:177
        n=qf_meas_bus_nr(i,2)
# wls.m:178
        for k in np.arange(1,nbus).reshape(-1):
            if k == m:
                g52[i,k]=np.dot(V_SE(n),(np.dot(G(m,n),sin(phi_SE(m) - phi_SE(n))) - np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n))))) + np.dot(np.dot(2,V_SE(m)),(B(m,n) - bbus(m,n)))
# wls.m:181
            else:
                if k == n:
                    g52[i,k]=np.dot(V_SE(m),(np.dot(G(m,n),sin(phi_SE(m) - phi_SE(n))) - np.dot(B(m,n),cos(phi_SE(m) - phi_SE(n)))))
# wls.m:183
                else:
                    g52[i,k]=0
# wls.m:185
    # g61 - Derivative of PMU Phases with respect to angles
    g61=zeros(n_phi_meas,nbus - 1)
# wls.m:191
    for i in np.arange(1,n_phi_meas).reshape(-1):
        m=pmu_meas_bus_nr(i,1)
# wls.m:193
        n=pmu_meas_bus_nr(i,2)
# wls.m:194
        for k in np.arange(1,(nbus - 1)).reshape(-1):
            if k + 1 == m:
                g61[i,k]=1
# wls.m:197
            if k + 1 == n:
                g61[i,k]=- 1
# wls.m:200
    # g62 - Derivative of PMU Phases with respect to V_SE
    g62=zeros(n_phi_meas,nbus)
# wls.m:205
    g=np.concat([[g11,g12],[g21,g22],[g31,g32],[g41,g42],[g51,g52],[g61,g62]])
# wls.m:207
    mmm,nnn=size(g,nargout=2)
# wls.m:214
    rk=rank(g)
# wls.m:215
    if rk < min(mmm,nnn):
        error('System is not observable')
    ## Hessian Matrix or Gain Matrix(double derivative), H
    H=np.dot(g.T,(numpy.linalg.solve(sigma_square,g)))
# wls.m:220
    ## Objective Function, J
# J = sum(sigma_square\res.^2); # sum(inv(sigma_square)*res.^2); #<FOR TIME SAVING CALCULATION>
## Residue
    res=y_perturbed - f
# wls.m:224
    delta_x=numpy.linalg.solve(H,(np.dot(g.T,(numpy.linalg.solve(sigma_square,res)))))
# wls.m:226
    state=state + delta_x
# wls.m:227
    phi_SE[np.arange(2,end())]=state(np.arange(1,nbus - 1))
# wls.m:228
    V_SE=state(np.arange(nbus,end()))
# wls.m:229
    normF=max(abs(delta_x))
# wls.m:231
    if normF < tol:
        converged=1
# wls.m:233
    # fprintf('WLS iteration # #4d: norm of mismatch: #5.20f\n', iters,normF);


## THEORY: Variance Covariance matrix of estimates sigma_x
sigma_x=diag(inv(np.dot(g.T,(numpy.linalg.solve(sigma_square,g)))))
# wls.m:238
sigma_x_v=sqrt(sigma_x(np.arange(nbus,end())))
# wls.m:239
sigma_x_phi=np.concat([[0],[np.dot(sqrt(sigma_x(np.arange(1,nbus - 1))),180) / pi]])
# wls.m:240