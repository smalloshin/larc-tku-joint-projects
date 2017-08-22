import networkx as nx
from yenKSP import *
import pickle

# return the dictionary of the edges of path p with their weights 
def get_edge_weights(g,p):
    edge_weights = dict()
    for i in range(len(p)-1):
       u = p[i]
       v = p[i+1]
       edge_weights[u+"_"+v]=float(g.edge[u][v]['weight'])
    return edge_weights


def get_network_graph(node_file,edge_file):
    nf = open(node_file,'r')
    ef = open(edge_file,'r')

    g = nx.DiGraph()

    od_list = []
    for line in ef:
        _,o,d,t = line.replace("\n","").split(",")
        od_list.append((o,d,float(t)))
    g.add_weighted_edges_from(od_list)

    # Write top-k shortest path (k=5 for now)
    od_ksp_dict = dict()
    od_asp_dict = dict() 
    import time
    pre_t = time.time()

    for o in g.nodes():
        for d in g.nodes():
            o = "79"
            d = "67"
            key = o +"_" +d
            # Compute top-k shortest paths for od
            cost_list, path_list = k_shortest_paths(g,o,d,k=5)            
            ksp_lists = []
            for i in range(len(path_list)):
                edges_dict = get_edge_weights(g,path_list[i])
                ksp_lists.append([cost_list[i],path_list[i],edges_dict])
            od_ksp_dict[key]=ksp_lists
            import pprint 
            pprint.pprint(ksp_lists)
            exit()

            """        
            # Compute 
            asp_lists = []
            for path in nx.all_simple_paths(g,"23","11",cutoff=28):
                edges_dict = get_edge_weights(g,path) 
                costs = sum([edges_dict[x] for x in edges_dict])
                asp_lists.append([costs,path,edges_dict])
            od_asp_dict[key]=asp_lists    
            """

    pickle.dump(od_ksp_dict,open('od_ksp_dict.pkl','w'))
    # pickle.dump(od_asp_dict,open('od_asp_dict.pkl','w'))
    print time.time()-pre_t

if __name__=="__main__":
    get_network_graph(node_file="nodes_mrt_nolrt.txt",edge_file="edges_mrt_nolrt.txt")
