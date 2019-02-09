import socket
import time
import OpenOPC
import os
import pandas
import numpy

from pypower import case14
import re


def read_data(opc):
    '''With an already open OPC client, read data for IEEE14 bus system, and return the P,Q injections and flow values
    '''
    ppc=case14.case14()
    tags=['Pmon4', 'Qmon4', 'Pmon5', 'Qmon5', 'Pmon9', 'Qmon9', 'Pmon10', 'Qmon10', 'Pmon11', 'Qmon11', 'Pmon12', 'Qmon12', 'Pmon13', 'Qmon13', 'Pmon14', 'Qmon14', 
    'P1', 'Q1', 'P8', 'Q8', 'Pinj2', 'Qinj2', 'Pinj3', 'Qinj3', 'Pinj6', 'Qinj6', 'PFLOW_1_2', 'QFLOW_1_2', 'PFLOW_1_5',
     'QFLOW_1_5', 'PFLOW_2_3', 'QFLOW_2_3', 'PFLOW_2_4', 'QFLOW_2_4', 'PFLOW_2_5', 'QFLOW_2_5', 'PFLOW_3_4',
    'QFLOW_3_4', 'PFLOW_4_5', 'QFLOW_4_5', 'PFLOW_6_11', 'QFLOW_6_11', 'PFLOW_6_12', 'QFLOW_6_12', 'PFLOW_6_13', 'QFLOW_6_13', 'PFLOW_9_10', 'QFLOW_9_10',
    'PFLOW_9_14', 'QFLOW_9_14', 'PFLOW_10_11', 'QFLOW_10_11', 'PFLOW_12_13', 'QFLOW_12_13', 'PFLOW_13_14', 'QFLOW_13_14']

    # read_tags=['OPAL_RT.OPAL_RT_device.OPAL_RT_tags.'+str(i) for i in tags]

    p=numpy.empty(len(ppc['bus']))
    p[:]=numpy.nan
    q=numpy.empty(len(ppc['bus']))
    q[:]=numpy.nan
    pij=numpy.empty(len(ppc['branch']))
    pij[:]=numpy.nan
    qij=numpy.empty(len(ppc['branch']))
    qij[:]=numpy.nan

    positive_inj_bus=[1,2,3,6,8]
    p[6]=0.0
    q[6]=0.0
    for tag in tags:
        if 'FLOW' in tag:
            continue
        val,_,_=opc.read('OPAL_RT.OPAL_RT_device.OPAL_RT_tags.'+tag)
        if tag[0]=='P':
            bus=int(re.findall('\d+',tag)[0])
            if bus in positive_inj_bus:
                p[bus-1]=val
            else:
                p[bus-1]=(-1)*val
                # load power drawn is given. Pinj should be negative in that case
        if tag[0]=='Q':
            bus=int(re.findall('\d+',tag)[0])
            if bus in positive_inj_bus:
                q[bus-1]=val
            else:
                q[bus-1]=(-1)*val
                # load power drawn is given. Pinj should be negative in that case
    for i,line in enumerate(ppc['branch']):
        print(i,line[0],line[1])
        val,_,_=opc.read('OPAL_RT.OPAL_RT_device.OPAL_RT_tags.PFLOW_'+str(int(line[0]))+'_'+str(int(line[1])))
        if val is None or val == 0:
            val=numpy.nan

        pij[i]=val

        val,_,_=opc.read('OPAL_RT.OPAL_RT_device.OPAL_RT_tags.QFLOW_'+str(int(line[0]))+'_'+str(int(line[1])))
        if val is None or val == 0:
            val=numpy.nan

        qij[i]=val
    return p,q,pij,qij
