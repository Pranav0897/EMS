import networkx as nx
import matplotlib.pyplot as plt
from pypower import case14
from networkx.drawing.layout import _sparse_fruchterman_reingold
import numpy
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
# !pip install pygraphviz
import pygraphviz as pgv


def draw_graph(Pline,Qline,Vbus,phibus,filename='graph.png'):
    G = nx.DiGraph()
    ppc=case14.case14() 
    cols=['green']

    for bus in ppc['bus']:
        busnum=int(bus[0])
        phi=phibus[busnum-1]*180.0/numpy.pi
        xlabel='V:'+str('%.2f'%round(Vbus[busnum-1],2))+'kV /_'+str("%.2f" % round(phi,2))
        G.add_node(busnum,color=cols[0],label=busnum,xlabel=xlabel)
    for i,line in enumerate(ppc['branch']):
        ptag=''
        if numpy.isnan(Pline[i]):
            ptag='NA'
        else:
            ptag=str('%.2f'%round(Pline[i]*100,2))
            qtag=str('%.2f'%round(numpy.abs(Qline[i])*100,2))
            if Qline[i]<0:
                label='Pline:'+ptag+'-j'+qtag+'MVA'
            else:
                label='Pline:'+ptag+'+j'+qtag+'MVA'
        G.add_edge(int(line[0]),int(line[1]),taillabel=label)
    G.graph['graph']={'rankdir':'TD','splines':'ortho','size': (10,10),'nodesep':2.5,'forcelabels':'True'}
    G.graph['node']={'shape':'line','width':0.1,'style':'filled'}
    G.graph['edges']={'arrowsize':'4.0','labelfontcolor':'red','labeldistance':5}

    A = to_agraph(G)
    print(A)
    A.layout('dot')
    A.draw(filename)
def test_draw_graph():
    Pline=[1.56882890532245,
    0.755103818256536,
    0.732375792384863,
    0.561314959394532,
    0.415162150181031,
    -0.232856900957646,
    -0.611582304445186,
    0.280741759163588,
    0.160797575831596,
    0.440873208602258,
    0.0735327698269188,
    0.0778606701532139,
    0.177479768622125,
    0,
    0.280741759163588,
    0.0522755246952551,
    0.0942638102999287,
    -0.0378532238134657,
    0.0161425777116624,
    0.0564385097994040]
    # for i in range(20):
    #     Pline.append(10*i**2)
    Vbus=[1.06000000000000,
    1.04500000000000,
    1.01000000000000,
    1.01767085369176,
    1.01951385981906,
    1.07000000000000,
    1.06151953249094,
    1.09000000000000,
    1.05593172063697,
    1.05098462499985,
    1.05690651854037,
    1.05518856319710,
    1.05038171362860,
    1.03552994585357]

    Qline=[
        -0.204042916842113,
    0.0385499114282339,
    0.0356020295073742,
    -0.0155035039835294,
    0.0117099785889661,
    0.0447311562544703,
    0.158236419344405,
    -0.210721239959393,
    -0.0657532220179535,
    -0.198163689183588,
    0.0356047297109479,
    0.0250341423702647,
    0.0721657538864191,
    -0.171629705111225,
    0.0577869056899889,
    0.0421913779793890,
    0.0361000623827139,
    -0.0161506292191982,
    0.00753959166264773,
    0.0174717378117202]
    phise=[0,
    -0.0869626030817508,
    -0.222094936468041,
    -0.179994118296443,
    -0.153132671974112,
    -0.248202389724111,
    -0.233169533202744,
    -0.233169532486174,
    -0.260726434585986,
    -0.263497445619242,
    -0.258145106644731,
    -0.263118642811912,
    -0.264526979983983,
    -0.279839946687294]
    # for i in range(14):
    #     Vbus.append(i*2*18)
    #     phise.append(i*(-1)**i/20.0)
    Vbus=[i*18 for i in Vbus]
    draw_graph(Pline,Qline,Vbus,phise,filename='graph1.png')

    
# fixed_positions = {1:(0,0),2:(40,0),3:(60,30),4:(50,30),5:(40,10),6:(30,10),9:(30,30),7:(40,40),8:(40,60),10:(10,20),11:(7,15),12:(-10,15),13:(-15,40),14:(-20,50)}#dict with two of the positions set
# fixed_nodes = fixed_positions.keys()
# pos=nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
# pos=nx.shell_layout(G)
# pos=nx.spring_layout(G,k=10, scale=100)
# pos=_sparse_fruchterman_reingold(nx.adjacency_matrix(G).astype(numpy.int64))
# print(pos)
# plt.figure()
# node_lable_dict={}
# for bus in ppc['bus']:
#     #change value here, to perhaps Voltage in KV

# for nodes in G.nodes():
#     nx.set_node_attributes(G,'name',node_lable_dict)
# nx.draw(G,pos,edge_color='black',width=1,linewidths=1,node_size=500,
#         node_color='pink',alpha=0.9,labels=node_lable_dict)
# edge_lable_dict={}
# for line in ppc['branch']:
#     key=(int(line[0]),int(line[1]))
#     value=int(line[0])*int(line[1])
#     edge_lable_dict[key]=value
# for edges in G.edges():
#     nx.set_edge_attributes(G,'weight',edge_lable_dict)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_lable_dict,font_color='red')
# plt.axis('off')
# plt.show()