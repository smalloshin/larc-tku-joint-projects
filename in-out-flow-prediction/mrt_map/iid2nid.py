import pickle


def iid2nid(nodefile="nodes_mrt_nolrt.txt"):
    f = open(nodefile,'r')
    
    # iid_dict: internal_id --> node_id 
    iid_dict = dict()
    for line in f:
        iid_list = line.replace("\n","").split(",")   
        
        nid = iid_list[0]
        n_iid = int(iid_list[6])

        if n_iid==0:
            print "[Warning] "+iid_list[2]+"has 0 internal id"
            continue

        for i in range(n_iid):
            iid_dict[iid_list[8+i]]=n_iid
    
    print "[Info] Save iid2nid_dict.pkl......"
    print "[Info] iid2nid_dict: internal_id-->node_id"
    pickle.dump(iid_dict,open('iid2nid_dict.pkl','w'))
    print "Done!"
        

if __name__=="__main__":
    iid2nid()
