import networkx as nx
import matplotlib.pyplot as plt
from pypower import case14

G = nx.DiGraph()
ppc=case14.case14() 
for bus in ppc['bus']:
    G.add_node(int(bus[0]))
for line in ppc['branch']:
    G.add_edge(int(line[0]),int(line[1]))

fixed_positions = {1:(0,0),2:(40,0),3:(60,30),4:(50,30),5:(40,10),6:(30,10),9:(30,30),7:(40,40),8:(40,60),10:(10,20),11:(7,15),12:(-10,15),13:(-15,40),14:(-20,50)}#dict with two of the positions set
fixed_nodes = fixed_positions.keys()
pos=nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
print(pos)
plt.figure()
node_lable_dict={}
for bus in ppc['bus']:
    #change value here, to perhaps Voltage in KV
    value=int(bus[0])
    node_lable_dict[int(bus[0])]=value

nx.draw(G,pos,edge_color='black',width=1,linewidths=1,node_size=500,
        node_color='pink',alpha=0.9,labels=node_lable_dict)
edge_lable_dict={}
for line in ppc['branch']:
    key=(int(line[0]),int(line[1]))
    value=int(line[0])*int(line[1])
    edge_lable_dict[key]=value
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_lable_dict,font_color='red')
plt.axis('off')
plt.show()
