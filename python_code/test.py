import numpy as np
import scipy.io as sio
from functions import *

case14 = sio.loadmat('case14.mat')['a_dict']
[busdatas, linedatas, gencost] = myformat(case14)

[ybus, A] = ybus_incidence(linedatas, busdatas)

# [V,phi] = newton(ybus,busdatas,linedatas)
[V, phi] = newton_alt(ybus, busdatas, linedatas)

[P,Q,Pij,Qij] = measurements(linedatas,V,phi,ybus)

# state_estimate(P,Q,Pij,Qij, case14)
# [vse, phise] = state_estimate_alt("case14.mat", P, Q, Pij, Qij)

# print("***************busdatas******************")
# print(busdatas)

# print("***************linedatas******************")
# print(linedatas)

# print("***************gencost******************")
# print(gencost)

# print("***************ybus******************")
# print(ybus)

# print("***************A******************")
# print(A)

# print("***************V******************")
# print(V)

# print("***************phi******************")
# print(phi)

# print("***************P******************")
# print(P)

# print("***************Q******************")
# print(Q)


print("***************Pij******************")
print(Pij)

print("***************Qij******************")
print(Qij)



# ideas
# 1. check J1 from matlab code and compare it to python output, see where is the overflow error coming from.
# 2. create a bash script to call python and matlab as needed.
