import pickle


def iid2nid(nodefile="nodes_mrt_nolrt.txt"):
    f = open(nodefile,'r')
    
    # iid_dict: internal_id --> node_id 
    iid_dict = dict()
    for line in f:
        iid_list = line.replace("\n","").split(",")   
        
        nid = iid_list[0]
        n_iid = int(iid_list[6])

        print "**********",iid_list[2],"**********"
        #iid start from the 8th column
        print n_iid
        for i in range(n_iid):
            print i,iid_list[8+i]
            iid_dict[iid_list[8+i]]=n_iid


if __name__=="__main__":
    iid2nid()
