# Generated with SMOP  0.41-beta
from . import *
import numpy as np # from numpy import *
# newton.m

def newton(Ybus_nr=None,*args,**kwargs):
    varargin = newton.varargin
    nargin = newton.nargin

    ## Newton-Raphson Load Flow
    global nbus,baseMVA,busdatas
    ## Getting busdata
    type_=busdatas(np.arange(),2)
# newton.m:6
    
    V=busdatas(np.arange(),3)
# newton.m:7
    
    Vsp=busdatas(np.arange(),3)
# newton.m:8
    
    phi=busdatas(np.arange(),4)
# newton.m:9
    
    Pg=busdatas(np.arange(),5) / baseMVA
# newton.m:10
    Qg=busdatas(np.arange(),6) / baseMVA
# newton.m:11
    Pl=busdatas(np.arange(),7) / baseMVA
# newton.m:12
    Ql=busdatas(np.arange(),8) / baseMVA
# newton.m:13
    Qmin=busdatas(np.arange(),9) / baseMVA
# newton.m:14
    
    Qmax=busdatas(np.arange(),10) / baseMVA
# newton.m:15
    
    Psp=Pg - Pl
# newton.m:16
    
    Qsp=Qg - Ql
# newton.m:17
    
    pq=find(type_ == 3)
# newton.m:18
    
    pv=find(type_ == logical_or(2,type_) == 1)
# newton.m:19
    
    npq=len(pq)
# newton.m:20
    
    G_nr=real(Ybus_nr)
# newton.m:21
    B_nr=imag(Ybus_nr)
# newton.m:22
    Tol=1
# newton.m:23
    itr=1
# newton.m:24
    ## Iteration Starts:
    while (Tol > 1e-09 and itr < 100):

        P=zeros(nbus,1)
# newton.m:27
        Q=zeros(nbus,1)
# newton.m:28
        for i in np.arange(1,nbus).reshape(-1):
            for k in np.arange(1,nbus).reshape(-1):
                P[i]=P(i) + np.dot(np.dot(V(i),V(k)),(np.dot(G_nr(i,k),cos(phi(i) - phi(k))) + np.dot(B_nr(i,k),sin(phi(i) - phi(k)))))
# newton.m:32
                Q[i]=Q(i) + np.dot(np.dot(V(i),V(k)),(np.dot(G_nr(i,k),sin(phi(i) - phi(k))) - np.dot(B_nr(i,k),cos(phi(i) - phi(k)))))
# newton.m:33
        # Checking Q-limit violations..
        if itr <= 7 and itr > 2:
            for n in np.arange(2,nbus).reshape(-1):
                if type_(n) == 2:
                    QG=Q(n) + Ql(n)
# newton.m:40
                    if QG < Qmin(n):
                        V[n]=V(n) + 0.01
# newton.m:42
                    else:
                        if QG > Qmax(n):
                            V[n]=V(n) - 0.01
# newton.m:44
        dP=Psp - P
# newton.m:49
        dQ1=Qsp - Q
# newton.m:50
        k=1
# newton.m:51
        dQ=zeros(npq,1)
# newton.m:52
        for i in np.arange(1,nbus).reshape(-1):
            if type_(i) == 3:
                dQ[k,1]=dQ1(i)
# newton.m:55
                k=k + 1
# newton.m:56
        r=np.concat([[dP(np.arange(2,nbus))],[dQ]])
# newton.m:59
        ## The Jacobian matrix
    # J1 - Derivative of Real Power Injections with Angles
        J1=zeros(nbus - 1,nbus - 1)
# newton.m:62
        for i in np.arange(1,(nbus - 1)).reshape(-1):
            m=i + 1
# newton.m:64
            for k in np.arange(1,(nbus - 1)).reshape(-1):
                n=k + 1
# newton.m:66
                if n == m:
                    for n in np.arange(1,nbus).reshape(-1):
                        J1[i,k]=J1(i,k) - np.dot(np.dot(V(m),V(n)),(np.dot(G_nr(m,n),sin(phi(m) - phi(n))) - np.dot(B_nr(m,n),cos(phi(m) - phi(n)))))
# newton.m:69
                    J1[i,k]=J1(i,k) - np.dot(V(m) ** 2,B_nr(m,m))
# newton.m:71
                else:
                    J1[i,k]=np.dot(np.dot(V(m),V(n)),(np.dot(G_nr(m,n),sin(phi(m) - phi(n))) - np.dot(B_nr(m,n),cos(phi(m) - phi(n)))))
# newton.m:73
        # J2 - Derivative of Real Power Injections with V
        J2=zeros(nbus - 1,npq)
# newton.m:78
        for i in np.arange(1,(nbus - 1)).reshape(-1):
            m=i + 1
# newton.m:80
            for k in np.arange(1,npq).reshape(-1):
                n=pq(k)
# newton.m:82
                if n == m:
                    for n in np.arange(1,nbus).reshape(-1):
                        J2[i,k]=J2(i,k) + np.dot(V(n),(np.dot(G_nr(m,n),cos(phi(m) - phi(n))) + np.dot(B_nr(m,n),sin(phi(m) - phi(n)))))
# newton.m:85
                    J2[i,k]=J2(i,k) + np.dot(V(m),G_nr(m,m))
# newton.m:87
                else:
                    J2[i,k]=np.dot(V(m),(np.dot(G_nr(m,n),cos(phi(m) - phi(n))) + np.dot(B_nr(m,n),sin(phi(m) - phi(n)))))
# newton.m:89
        # J3 - Derivative of Reactive Power Injections with Angles
        J3=zeros(npq,nbus - 1)
# newton.m:94
        for i in np.arange(1,npq).reshape(-1):
            m=pq(i)
# newton.m:96
            for k in np.arange(1,(nbus - 1)).reshape(-1):
                n=k + 1
# newton.m:98
                if n == m:
                    for n in np.arange(1,nbus).reshape(-1):
                        J3[i,k]=J3(i,k) + np.dot(np.dot(V(m),V(n)),(np.dot(G_nr(m,n),cos(phi(m) - phi(n))) + np.dot(B_nr(m,n),sin(phi(m) - phi(n)))))
# newton.m:101
                    J3[i,k]=J3(i,k) - np.dot(V(m) ** 2,G_nr(m,m))
# newton.m:103
                else:
                    J3[i,k]=np.dot(np.dot(V(m),V(n)),(np.dot(- G_nr(m,n),cos(phi(m) - phi(n))) - np.dot(B_nr(m,n),sin(phi(m) - phi(n)))))
# newton.m:105
        # J4 - Derivative of Reactive Power Injections with V
        J4=zeros(npq,npq)
# newton.m:110
        for i in np.arange(1,npq).reshape(-1):
            m=pq(i)
# newton.m:112
            for k in np.arange(1,npq).reshape(-1):
                n=pq(k)
# newton.m:114
                if n == m:
                    for n in np.arange(1,nbus).reshape(-1):
                        J4[i,k]=J4(i,k) + np.dot(V(n),(np.dot(G_nr(m,n),sin(phi(m) - phi(n))) - np.dot(B_nr(m,n),cos(phi(m) - phi(n)))))
# newton.m:117
                    J4[i,k]=J4(i,k) - np.dot(V(m),B_nr(m,m))
# newton.m:119
                else:
                    J4[i,k]=np.dot(V(m),(np.dot(G_nr(m,n),sin(phi(m) - phi(n))) - np.dot(B_nr(m,n),cos(phi(m) - phi(n)))))
# newton.m:121
        J=np.concat([[J1,J2],[J3,J4]])
# newton.m:125
        X=numpy.linalg.solve(J,r)
# newton.m:127
        dTh=X(np.arange(1,nbus - 1))
# newton.m:128
        dV=X(np.arange(nbus,end()))
# newton.m:129
        ## record the phita V and phita Angle
        dV_sq[itr,np.arange()]=dV.T
# newton.m:131
        dTh_sq[itr,np.arange()]=dTh.T
# newton.m:132
        phi[np.arange(2,nbus)]=dTh + phi(np.arange(2,nbus))
# newton.m:134
        k=1
# newton.m:135
        for i in np.arange(1,nbus).reshape(-1):
            if type_(i) == 3:
                V[i]=dV(k) + V(i)
# newton.m:138
                k=k + 1
# newton.m:139
            else:
                V[i]=Vsp(i)
# newton.m:141
        itr=itr + 1
# newton.m:144
        Tol=max(abs(r))
# newton.m:145

    
    # fprintf('N-R Iterations = #4d', itr);fprintf('\n');
## Figure of convergence
# figure
# plot([1:itr-1],diag(dV_sq*dV_sq'),[1:itr-1],diag(dTh_sq*dTh_sq'),'g');
# title('Load Flow: phita-V and phita-Phi decrease according toIterations'); xlabel('iteration'); ylabel('phita V & Angle'); grid on #figure for convergence
    return V,phi
    
if __name__ == '__main__':
    pass
    