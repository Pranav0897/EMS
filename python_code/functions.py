import numpy as np
import matlabopti

def ybus_incidence(linedatas, busdatas):

	fb = linedatas[:, 0]-1
	tb = linedatas[:, 1]-1
	r = linedatas[:, 2]
	x = linedatas[:, 3]
	b = linedatas[:, 4]
	tap = linedatas[:, 5]

	nbus = busdatas[:, 0].shape[0]
	nbranch = linedatas[:, 0].shape[0]
	baseMVA = 100

	GS = busdatas[:,10]; # shunt conductance (MW at V = 1.0 p.u.)
	BS = busdatas[:,11]; # shunt susceptance (MW at V = 1.0 p.u.)

	Ysh = (GS + 1j*BS)/baseMVA
	Z = r + 1j*x
	Y = np.reciprocal(Z)

	A = np.zeros((nbranch+nbus, nbus))

	for i in range(nbus):
		for j in range(nbus):
			if(i==j): A[i, j] = 1

	for i in range(nbus, nbus+nbranch):
		A[i, int(fb[i-nbus])] = 1
		A[i, int(tb[i-nbus])] = -1

	Yprimitive = np.zeros((nbranch+nbus, ), dtype=complex)

	for i in range(nbranch):
	    Yprimitive[int(fb[i])] = Yprimitive[int(fb[i])] + 1j*b[i]/2 + (1-tap[i])*Y[i]/tap[i]**2
	    Yprimitive[int(tb[i])] = Yprimitive[int(tb[i])] + 1j*b[i]/2 + (tap[i]-1)*Y[i]/tap[i]

	Yprimitive[:nbus] = Yprimitive[:nbus] + Ysh;

	for i in range(nbranch):
		Yprimitive[i+nbus] = Y[i]/tap[i]

	return [np.dot(np.dot(np.transpose(A), np.diag(Yprimitive)), A), A]

def myformat(case14):
	bus = case14[0][0][2]
	gen = case14[0][0][3]
	branch = case14[0][0][4]
	gencost = case14[0][0][5]

	busdatas = np.zeros((bus.shape[0], bus.shape[1]-1))
	busdatas[:, :2] = bus[:, :2]
	for i in range(bus.shape[0]):
		if(busdatas[i, 1]==3): busdatas[i, 1] = 0
		if(busdatas[i, 1]==1): busdatas[i, 1] = 3
		if(busdatas[i, 1]==0): busdatas[i, 1] = 1

	busdatas[:, 2] = bus[:, 7]

	busdatas[:,3:12]=np.zeros((bus.shape[0],9))
	
	for i in range(gen.shape[0]):
		busdatas[int(gen[i, 0])-1, 4] = gen[i, 1]
		busdatas[int(gen[i, 0])-1, 5] = gen[i, 2]
	
	busdatas[:,6:8]=bus[:,2:4]
	
	for i in range(gen.shape[0]):
		busdatas[int(gen[i, 0])-1, 8] = gen[i, 4]
		busdatas[int(gen[i, 0])-1, 9] = gen[i, 3]

	busdatas[:,10:12]=bus[:, 4:6]

	temp = np.reshape(branch[:,8], branch[:,8].shape+(1,))
	linedatas = np.hstack((branch[:,:5],temp))
	
	for i in range(linedatas.shape[0]):
		if(linedatas[i, 5]==0): linedatas[i, 5] = 1

	temp1 = np.reshape(gen[:,0], gen[:,0].shape+(1,))
	temp2 = np.reshape(gen[:,9], gen[:,9].shape+(1,))
	temp3 = np.reshape(gen[:,8], gen[:,8].shape+(1,))
	gencost = np.hstack((temp1, gencost[:,4:7], temp2, temp3))
	
	return [busdatas, linedatas, gencost]

def newton(ybus, busdatas, linedatas):

	nbus = busdatas.shape[0]
	baseMVA = 100

	# Type of Bus 1-Slack, 2-PV, 3-PQ
	bus_type = busdatas[:, 1]
	
	# Slack Voltage and Voltage mag intitials
	V = busdatas[:, 2]
	
	# Slack Voltage and Voltage mag intitials
	Vsp = busdatas[:, 2]
	
	# Voltage Angle intitials
	phi = busdatas[:, 3]
	
	Pg = busdatas[:, 4]/baseMVA
	
	Qg = busdatas[:, 5]/baseMVA
	
	Pl = busdatas[:, 6]/baseMVA
	
	Ql = busdatas[:, 7]/baseMVA
	
	# Minimum Reactive Power Limit..
	Qmin = busdatas[:, 8]/baseMVA
	
	# Maximum Reactive Power Limit..
	Qmax = busdatas[:, 9]/baseMVA
	
	# calculate powers in the busses: P Specified
	Psp = Pg-Pl
	
	# calculate powers in the busses: Q Specified
	Qsp = Qg-Ql

	pq = []; pv = []
	for i, bus in enumerate(bus_type):
		if(bus==3): pq.append(i) #PQ Buses(there is no generation)
		if(bus==2 or bus==1): pv.append(i) #PV Buses

	# No. of PQ buses
	npq = len(pq)

	G_nr = ybus.real
	B_nr = ybus.imag
	Tol = 1
	itr = 1

	# iteration starts
	while(Tol>1e-9 and itr<100):
		P = np.zeros((nbus, ))
		Q = np.zeros((nbus, ))

		for i in range(nbus):
			for k in range(nbus):
				P[i] = P[i] + V[i]*V[k]*(G_nr[i,k]*np.cos(phi[i]-phi[k]) + B_nr[i,k]*np.sin(phi[i]-phi[k])) # pp. 77 Wang. (eq 2.9)
				Q[i] = Q[i] + V[i]*V[k]*(G_nr[i,k]*np.sin(phi[i]-phi[k]) - B_nr[i,k]*np.cos(phi[i]-phi[k]))

	    # checking Q-limit violations
		if(itr<=7 and itr>2):
			for n in range(1, nbus):
				if(bus_type[n]==2):
					QG = Q[n]+Ql[n]
					if(QG<Qmin[n]):
						V[n] = V[n] + 0.01
					elif(QG>Qmax[n]):
						V[n] = V[n] - 0.01

		dP = Psp-P  # Calculate change from specified value pp.78 Wang. (eq2.13)
		dQ1 = Qsp-Q 
		k = 0
		dQ = np.zeros((npq,))
		for i in range(nbus):
			if(bus_type[i] == 3):
				dQ[k] = dQ1[i]
				k = k+1

		r = np.hstack((dP[1:], dQ)) # Mismatch Vector, not considering the first value that is the slack bus P,Q

	    # The Jacobian Matrix
	    
	    # J1 - Derivative of Real Power Injections with Angles
		J1 = np.zeros((nbus-1, nbus-1), dtype=np.complex)
		for i in range(nbus-1):
			m = i+1
			for k in range(nbus-1):
				n=k+1
				if(n==m):
					for n in range(nbus):
						J1[i,k] = J1[i,k] - V[m]* V[n]*(G_nr[m,n]*np.sin(phi[m]-phi[n]) - B_nr[m,n]*np.cos(phi[m]-phi[n])) # pp. 84 Wang. eq(2.41)

					print(J1[i,k])
					print(V[m])
					print(2*B_nr[m,m])
					J1[i,k] = J1[i,k] - np.power(V[m], (2*B_nr[m,m]), dtype=np.complex)
					print("=")
					print(np.power(V[m], (2*B_nr[m,m]), dtype=np.complex))
					print("************************")
				else:
					J1[i,k] = V[m]* V[n]*(G_nr[m,n]*np.sin(phi[m]-phi[n]) - B_nr[m,n]*np.cos(phi[m]-phi[n])) # pp. 84 Wang. eq(2.42)

	    #J2 - Derivative of Real Power Injections with V
		J2 = np.zeros((nbus-1,npq), dtype=np.complex)
		for i in range(nbus-1):
			m = i+1
			for k in range(npq):
				n = pq[k]
				if(n==m):
					for n in range(nbus):
						J2[i,k] = J2[i,k] + V[n]*(G_nr[m,n]*np.cos(phi[m]- phi[n]) + B_nr[m,n]*np.sin(phi[m]-phi[n]))
					J2[i,k] = J2[i,k] + V[m]*G_nr[m,m]
				else:
					J2[i,k] = V[m]*(G_nr[m,n]*np.cos(phi[m]-phi[n]) + B_nr[m,n]*np.sin(phi[m]-phi[n]));
	    
        # J3 - Derivative of Reactive Power Injections with Angles
		J3 = np.zeros((npq,nbus-1), dtype=np.complex)
		for i in range(npq):
			m = pq[i]
			for k in range(nbus-1):
				n = k+1
				if (n == m):
					for n in range(nbus):
						J3[i,k] = J3[i,k] + V[m]*V[n]*(G_nr[m,n]*np.cos(phi[m]-phi[n]) + B_nr[m,n]*np.sin(phi[m]-phi[n]))
					J3[i,k] = J3[i,k] - np.power(V[m], (2*G_nr[m,m]), dtype=np.complex)
				else:
					J3[i,k] = V[m]* V[n]*(-G_nr[m,n]*np.cos(phi[m]-phi[n]) - B_nr[m,n]*np.sin(phi[m]-phi[n])) # pp. 84 Wang. eq(2.44) !!!!! MANFI
  
	    # J4 - Derivative of Reactive Power Injections with V
		J4 = np.zeros((npq,npq), dtype=np.complex)
		for i in range(npq):
			m = pq[i]
			for k in range(npq):
				n = pq[k]
				if(n==m):
					for n in range(nbus):
						J4[i,k] = J4[i,k] + V[n]*(G_nr[m,n]*np.sin(phi[m]- phi[n]) - B_nr[m,n]*np.cos(phi[m]-phi[n]))
					J4[i,k] = J4[i,k] - V[m]*B_nr[m,m]
				else:
					J4[i,k] = V[m]*(G_nr[m,n]*np.sin(phi[m]-phi[n]) - B_nr[m,n]*np.cos(phi[m]-phi[n])) # pp. 85 Wang. eq(2.48) !!!!! MANFI
	    
		J = np.vstack((np.hstack((J1, J2)), np.hstack((J3, J4))))
		# J = J.astype(np.float64)
		X = np.linalg.solve(J, r) # not sure, check. 
		dTh = np.real(X[:nbus-1]) # change in voltage angle
		dV = np.real(X[nbus-1:]) # change in voltage magnitude 

		dV_sq = np.array([])
		dV_sq = np.hstack((dV_sq, dV))

		dTh_sq = np.array([])
		dTh_sq = np.hstack((dTh_sq, dTh))

		# updating state vectors
		phi[1:] = dTh + phi[1:] # angle update

		k = 0
		for i in range(nbus):
			if(bus_type[i]==3):
				V[i] = dV[k]+V[i] # voltage magnitude update
				k=k+1
		else:
			V[i] = Vsp[i] # reset the slack and PV bus voltage to specified values

		itr = itr+1
		Tol = np.amax(np.absolute(r))

	return [V, phi]

def measurements(linedatas, V, phi, ybus):

	# power injection measurements
	nbus = int(max(max(linedatas[:, 0]), max(linedatas[:, 1])))
	nbranch = int(linedatas[:, 0].shape[0])
	b = linedatas[:, 4]
	fb = linedatas[:, 0]
	tb = linedatas[:, 1]
	G = np.real(ybus)
	B = np.imag(ybus)
	P = np.zeros(nbus, )
	Q = np.zeros(nbus, )

	for l in range(nbus):
		for m in range(nbus):
			P[l] = P[l] + V[l]* V[m]*(G[l,m]*np.cos(phi[l]-phi[m]) + B[l,m]*np.sin(phi[l]-phi[m]))
			Q[l] = Q[l] + V[l]* V[m]*(G[l,m]*np.sin(phi[l]-phi[m]) - B[l,m]*np.cos(phi[l]-phi[m])) 

	bbus_perturbed = np.zeros((nbus, nbus))
	for k in range(nbranch):
		bbus_perturbed[int(fb[k])-1, int(tb[k])-1] = b[k]/2
		bbus_perturbed[int(tb[k])-1, int(fb[k])-1] = bbus_perturbed[int(fb[k])-1, int(tb[k])-1]

	# power flows calculation
	Pij = np.zeros(nbranch, )
	Qij = np.zeros(nbranch, )
	Pji = np.zeros(nbranch, )
	Qji = np.zeros(nbranch, )
	for o in range(nbranch):
		m = int(fb[o])-1
		n = int(tb[o])-1
		Pij[o] = -V[m]**(2*G[m,n]) + V[m]*V[n]*(G[m,n]*np.cos(phi[m]-phi[n])+ B[m,n]*np.sin(phi[m]-phi[n]))
		Qij[o] = V[m]**(2*(B[m,n]- bbus_perturbed[m,n])) +V[m]*V[n]*(G[m,n]*np.sin(phi[m]-phi[n]) - B[m,n]*np.cos(phi[m]-phi[n]))
		Pji[o] = -V[n]**(2*G[n,m]) + V[n]*V[m]*(G[n,m]*np.cos(phi[n]-phi[m])+ B[n,m]*np.sin(phi[n]-phi[m]))
		Qji[o] = V[n]**(2*(B[n,m]- bbus_perturbed[n,m])) +V[n]*V[m]*(G[n,m]*np.sin(phi[n]-phi[m]) - B[n,m]*np.cos(phi[n]-phi[m]))

	return [P,Q,Pij,Qij]

def nlconse(h, P, Q, Pij, Qij, linedatas, ybus):

	nbus = max(max(linedatas[:,0]),max(linedatas[:,1]))
	nbranch = int(linedatas[:, 0].shape[0])
	Vse = h[P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]+nbus]
	phise = h[P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]+nbus:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]+2*nbus]
	b = linedatas[:, 4]
	fb = linedatas[:, 0]
	tb = linedatas[:, 1]
	G = np.real(ybus)
	B = np.imag(ybus)
	Pcal = np.zeros(nbus,)
	Qcal = np.zeros(nbus,)

	for l in range(nbus):
		for m in range(nbus):
			Pcal[l] = Pcal[l] + Vse[l]* Vse[m]*(G[l,m]*np.cos(phise[l]-phise[m]) + B[l,m]*np.sin(phise[l]-phise[m]))
			Qcal[l] = Qcal[l] + Vse[l]* Vse[m]*(G[l,m]*np.sin(phise[l]-phise[m]) - B[l,m]*np.cos(phise[l]-phise[m])) 

	bbus_perturbed = np.zeros((nbus,nbus))
	for k in range(nbranch):
		bbus_perturbed[int(fb[k])-1, int(tb[k])-1] = b[k]/2
		bbus_perturbed[int(tb[k])-1, int(fb[k])-1] = bbus_perturbed[int(fb[k])-1, int(tb[k])-1]

	# power flows calculation
	for o in range(nbranch):
		m = int(fb(o)-1)
		n = int(tb(o)-1)
		Pijcal[o] = -Vse[m]**(2*G[m,n]) + Vse[m]*Vse[n]*(G[m,n]*np.cos(phise[m]-phise[n])+ B[m,n]*np.sin(phise[m]-phise[n]))
		Qijcal[o] = Vse[m]**(2*(B[m,n]- bbus_perturbed[m,n])) +Vse[m]*Vse[n]*(G[m,n]*np.sin(phise[m]-phise[n]) - B[m,n]*np.cos(phise[m]-phise[n]))
	        
	ceq = np.hstack((Pcal-P-h[:P.shape[0]], Qcal-Q-h[P.shape[0]:P.shape[0]+Q.shape[0]], Pijcal-Pij-h[P.shape[0]+Q.shape[0]:P.shape[0]+Q.shape[0]+Pij.shape[0]], Qijcal-Qij-h[P.shape[0]+Q.shape[0]+Pij.shape[0]:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]]))

	c = np.array([])

	return [c, ceq]

def objse(h, P, Q, Pij, Qij):

	return np.sum(np.multiply(h[:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]], h[1:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]]))

def state_estimate(P,Q,Pij,Qij, casefile):

	[busdatas, linedatas, gencost] = myformat(casefile)
	nbus = int(max(max(linedatas[:,0]),max(linedatas[:,1])))

	# %gencost(:,2)=0;
	# %% Ybus calculation

	[ybus, A] = ybus_incidence(linedatas,busdatas);

	# Measurements
	# [V,phi] = newton(ybus,busdatas,linedatas);
	# [P,Q,Pij,Qij] = measurements(linedatas,V,phi,ybus);
	# P[:2] = P[:2]*2

	# SE initialization
	errorlb = -np.inf*np.ones(P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0],)
	errorub = np.inf*np.ones(P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0],)
	V_estub = 1.1*np.ones(P.shape[0],);
	V_estlb = 0.9*np.ones(P.shape[0],);
	phi_estub = (np.pi)*np.ones(P.shape[0],);
	phi_estlb = -(np.pi)*np.ones(P.shape[0],);
	phi_estlb[0] = 0;
	phi_estub[0] = 0;
	
	P = P.tolist()
	P = [0 for x in range(len(P))] 
	Q = Q.tolist() 
	Q = [0 for x in range(len(Q))]
	Pij = Pij.tolist() 
	Pij = [0 for x in range(len(Pij))]
	Qij = Qij.tolist()
	Qij = [0 for x in range(len(Qij))] 
	linedatas = linedatas.tolist()
	# print(linedatas)
	# quit() 
	ybus_real = np.real(ybus)
	ybus_imag = np.imag(ybus)	
	ybus_real = ybus_real.tolist()
	ybus_imag = ybus_imag.tolist()

	opt = matlabopti.initialize()
	# [x_old, val] = opt.myfmincon(nbus, P, Q, Pij, Qij, linedatas, ybus_real, ybus_imag)

	# Vse = x_old[P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]+nbus]
	# phise = x_old[P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]+nbus:P.shape[0]+Q.shape[0]+Pij.shape[0]+Qij.shape[0]+2*nbus]

	fuck = opt.myfmincon(nbus, P, Q, Pij, Qij, linedatas, ybus_real, ybus_imag)

	print(fuck)

	# return [Vse, phise]
	return None


# ideas
# 1. check J1 from matlab code and compare it to python output, see where is the overflow error coming from.