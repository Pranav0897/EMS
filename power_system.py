import numpy as np
import copy
from busdata import busdata
from linedata import get_line_data
class power_system(busdata,linedata):
    global test_case,baseMVA,V_true,Phi_true, nbus,busdatas
    global fb,tb,nbus,nbranch,b_nominal #from power_flow
    global nbus,bus_type,Pgen,Qgen,Qload,Pload #from power_inj
    global fb,tb,nbranch,nbus,linedatas,baseMVA #from ybus_incidence
    def get_bus_data(self):
        return 0
    def get_line_data(self):
        return 0
    def power_flow(self):
        return 0
    def power_inj(self):
        return 0
    def ybus_incidence(self,r=None,x=None,b=None):
        tap=self.linedatas[:,5]
        # GS=busdatas(np.arange(),11)
        # BS=busdatas(np.arange(),12)
        GS=self.busdatas[:,10]
        BS=self.busdatas[:,11]
        
        Ysh=(GS + np.dot(1j,BS)) / self.baseMVA
        
        Z=r + np.dot(1j,x)
        
        Y=1.0 / Z
        ## Formation of Bus Incidence matrix A (signs: comes in is -1, goes out is +1)
        A=np.zeros(shape=(self.nbranch + self.nbus,self.nbus),dtype=np.complex_)
        for i in np.arange(1,self.nbus).reshape(-1):
            for j in np.arange(1,self.nbus).reshape(-1):
                if (i == j):
                    A[i,i]=1
        
        for i in np.arange(self.nbus + 1,self.nbus + self.nbranch).reshape(-1):
            A[i,self.fb(i - self.nbus)]=1
            A[i,self.tb(i - self.nbus)]=- 1
        
        ## Calculation of primitive matrix
        Yprimitive=np.zeros(shape=(self.nbranch + self.nbus,1),dtype=np.complex_)

        # For buses:
        for i in np.arange(1,self.nbranch).reshape(-1):
            Yprimitive[self.fb(i)]=Yprimitive(self.fb(i)) + np.dot(1j,b(i)) / 2 + np.dot((1 - tap(i)),Y(i)) / tap(i) ** 2
            Yprimitive[self.tb(i)]=Yprimitive(self.tb(i)) + np.dot(1j,b(i)) / 2 + np.dot((tap(i) - 1),Y(i)) / tap(i)
        
        Yprimitive[np.arange(1,self.nbus)]=Yprimitive(np.arange(1,self.nbus)) + Ysh
    
        # Branches:
        for i in np.arange(1,self.nbranch).reshape(-1):
            Yprimitive[i + self.nbus]=Y(i) / tap(i)
        ## Bus Admittance matrix:
        Ybus=np.dot(np.dot(A.T,np.diag(Yprimitive)),A)

        return Ybus,A

    def test(self):
        self.test_case=14
        self.MC_tests=10
        self.display_SE_results=1
        self.Ps=0
        self.Pm=0
        ## Variance of Voltage magnitude, Active and reactive Power Injections and flows:
        sigma_v=0.01
        sigma_phi=0.01
        sigma_pi=0.01
        sigma_qi=0.01
        sigma_pf=0.01
        sigma_qf=0.01
        rho=0.0

        self.get_line_data()
        # TODO do proper conversion
        # self.fb=self.linedatas(np.arange(),1)
        # self.tb=self.linedatas(np.arange(),2)
        # r_nominal=linedatas(np.arange(),3)
        # x_nominal=linedatas(np.arange(),4)
        # b_nominal=linedatas(np.arange(),5)

        self.fb=self.linedatas[:,0]
        self.tb=self.linedatas[:,1]
        r_nominal=self.linedatas[:,2]
        x_nominal=self.linedatas[:,3]
        b_nominal=self.linedatas[:,4]
        nbranch=len(self.fb)

        self.nbus=copy.deepcopy(self.test_case)

        negatives=0
        # TODO do proper conversion
        

        self.get_bus_data()

        # bus_type=busdatas(np.arange(),2)
        # Pgen=busdatas(np.arange(),5) / baseMVA
        # Qgen=busdatas(np.arange(),6) / baseMVA
        # Pload=busdatas(np.arange(),7) / baseMVA
        # Qload=busdatas(np.arange(),8) / baseMVA

        bus_type=self.busdatas[:,1]
        Pgen=self.busdatas[:,4] / self.baseMVA
        Qgen=self.busdatas[:,5] / self.baseMVA
        Pload=self.busdatas[:,6] / self.baseMVA
        Qload=self.busdatas[:,7] / self.baseMVA

        Ybus_nominal,A_incidence=self.ybus_incidence(r_nominal,x_nominal,b_nominal)
        
        G=np.real(Ybus_nominal)
        B=np.imag(Ybus_nominal)
        V_error_seq=[]
        phi_error_seq=[]
        V_SE_seq=[]
        phi_SE_seq=[]
        V_nr_seq=[]
        phi_nr_seq=[]
        sigma_x_v_seq=[]
        sigma_x_phi_seq=[]

        
        V_act,phi_act=self.newton(Ybus_nominal,nargout=2)
        V_nr=copy.deepcopy(V_act)
        phi_nr=copy.deepcopy(phi_act)

        for MC_test in np.arange(1,MC_tests).reshape(-1):
            # V_nr = np.random.normal(V_nr, 0.001 .* V_nr);
            #  phi_nr = np.random.normal(phi_nr, 0.02 .* phi_nr);
            phi_nr_dg=np.dot(180 / np.pi,phi_nr)
            ## WLS State Estimation
            # Shunt Admittance Matrix Formation: # Off-diagonals are the mutual admittances between the respective nodes
            bbus=np.zeros((self.nbus,self.nbus))
            for k in np.arange(1,nbranch).reshape(-1):
                bbus[self.fb(k),self.tb(k)]=b_nominal(k) / 2
                bbus[self.tb(k),self.fb(k)]=bbus(self.fb(k),self.tb(k))
            meas_calc
            wls
            phi_SE_dg=np.dot(180 / np.pi,phi_SE)
            ## Collecting V and Phi errors in each iterarion
            V_error=V_nr - V_SE
            phi_error=phi_nr_dg - phi_SE_dg
            ## Display the WLS results
        # if test_case >= 57 || mod(MC_test,20) == 0
        #     clc
            if display_SE_results:
                disp('_________________State Estimation_________________')
                disp('Bus V_act V_SE V_NR V_Er Ph_act Ph_SE Ph_NR Ph_Er')
                for m in np.arange(1,self.nbus).reshape(-1):
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

