# Generated with SMOP  0.41-beta
from . import *
# from numpy import *
import numpy as np
import copy

## Measurement Data Preparation:
#states = 27
#measurements = 1.5 * states = 41
## The bus numbers that traditional voltage mag. measurement happens:
#v_meas_bus_nr = [1 2 3 6 8]';
v_meas_bus_nr=np.concatenate([1,2,3,6,8]).T
## The bus numbers that PMU measurement happens:
pmu_meas_bus_nr=[]
v_meas_bus_nr=np.unique(np.concatenate([[np.ravel(pmu_meas_bus_nr)],[v_meas_bus_nr]]))
## The bus numbers that active power injection measurement happens:
pi_meas_bus_nr=np.concatenate([1,2,6,8,9,10,11,12,14]).T
## The bus numbers that active and reactive power injection measurement happens:
qi_meas_bus_nr=copy.deepcopy(pi_meas_bus_nr)
## The bus numbers that active power flow measurement happens: (Power is flowing from the first number to the second)
pf_meas_bus_nr=[[1,2],
                [1,5],
                [2,3],
                [2,4],
                [3,4],
                [4,5],
                [4,7],
                [5,6],
                [6,13]]
## The bus numbers that reactive power flow measurement happens:
qf_meas_bus_nr=copy.deepcopy(pf_meas_bus_nr)