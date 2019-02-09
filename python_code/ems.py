import socket
import time
import OpenOPC
import os
import pandas
import numpy

from pypower import case14
import re

def rw_data_plot_graph():
    base='C:\\Users\\Administrator\\Downloads\\akash_pranav2\\final_data'

    TCP_IP = '127.0.0.1'
    TCP_PORT = 4575
    BUFFER_SIZE = 1024

    load_bus=[2,3,4,5,6,9,10,11,12,13,14]
    gen_bus=[1, 2, 3, 6, 8]

    # print(load_bus)
    pload_tags=['Pset'+str(i) for i in load_bus]
    qload_tags=['Qset'+str(i) for i in load_bus]
    rtds_tags=[]
    rtds_tags.extend(pload_tags)
    rtds_tags.extend(qload_tags)
    rtds_tags.append('P_BUS2x1')

    nyiso_tags=[str(i) for i in load_bus]
    nyiso_tags.extend(['q'+str(i) for i in load_bus])
    nyiso_tags.append('P_BUS2x1')

    nyiso2rtds=dict(zip(nyiso_tags,rtds_tags))
    # print(nyiso2rtds)

    # read_tags=['P'+str(i) for i in gen_bus]
    # read_tags.extend(['Q'+str(i) for i in gen_bus])
    read_tags=[]


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))

    opc=OpenOPC.open_client('localhost')
    opc.connect('Kepware.KEPServerEX.V6','localhost')

    for file in os.listdir(base):
        df=pandas.read_csv(os.path.join(base,file))
        rows=[]
        for i in df.index:
            row_dict={}
            for k in df.keys():
                row_dict[k]=df[k][i]
            '''
            write data:
            '''
            for k in nyiso_tags:
                val2write=numpy.array(df[k][i]).tolist()
                if k=='P_BUS2x1':
                    str1='SetSlider "Subsystem #1 : CTLs : Inputs : P_BUS2x1" = '+str(val2write)+';'
                else:
                    bus=int((filter(str.isdigit, nyiso2rtds[k])))
                    cm="Subsystem #1 : Loads : DL"+str(bus)+" : "+str(nyiso2rtds[k])
                    str1='SetSlider "'+str(cm)+'" = '+str(val2write)+';' 
                print str1
                s.send(str1)
                s.send('SUSPEND 0.2;')
                s.send('ListenOnPortHandshake(temp_string);')
                tokenstring = s.recv(BUFFER_SIZE)
                # print "The token string returned is: ", tokenstring            
            '''
            Wait for system to stabilise
            '''
            time.sleep(10)
            
            '''
            read data
            '''
            p,q,pij,qij=read_data(opc)
    #         for tags in read_tags:
    #             val,_,_=opc.read('OPAL_RT.OPAL_RT_device.OPAL_RT_tags.'+str(tags))
    #             row_dict[tags]=val
    #             print("Received: ",tags,val)
            v,phi=eng.state_estimate('case14.mat',p,q,pij,qij)
            draw_graph(pij,qij,v,phi,filename=os.path.join(base,'graph1.png'))
            break
        break


    s.send('ClosePort(%d);' % (TCP_PORT))
    s.close()

    print 'All iterations complete.'
    return os.path.join(base,'graph1.png')
# finish = time.clock() - start
# print 'The execution time is %fs.' %finish
