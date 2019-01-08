import numpy as np
import scipy.io as sio
from functions import *

case14 = sio.loadmat('case14.mat')['a_dict']
[busdatas, linedatas, gencost] = myformat(case14)
[ybus, A] = ybus_incidence(linedatas, busdatas)
[V,phi] = newton(ybus,busdatas,linedatas)
[P,Q,Pij,Qij] = measurements(linedatas,V,phi,ybus)
[Vse,phise]= state_estimate(P,Q,Pij,Qij, case14)


# print(busdatas)
# print(A)