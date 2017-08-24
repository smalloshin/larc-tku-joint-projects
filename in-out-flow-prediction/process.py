import pandas as pd
import numpy as np
from sklearn import mixture
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')


from matplotlib import pyplot as plt
from pprint import *
import smrt_util as smu
import cPickle as Pickle


# decide how many gaussian to fit data
def gmm_selection(X,max_components):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, max_components)

    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(random_state=1,n_components=n_components)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_cvtype = cv_type
    return best_gmm, best_cvtype

def gmm_clustering(X,cv_type="full",n_components=2):
    gmm = mixture.GaussianMixture(covariance_type=cv_type,n_components=n_components)
    gmm.fit(X)
    clusters = gmm.predict(X)
    return clusters

# 06:00 (21600th sec) ~ 24:00 (86400th sec)
# slice_len: in seconds
def init_inout_map(df,slice_len=300, start_tick=21600, end_tick=86400, n_station=104):
    in_map = dict()
    out_map = dict()
    n_slice = int((end_tick-start_tick)/slice_len)
   
    # +1 --> index for outside 
    for i in range(n_slice):
        in_map[i]=np.zeros((n_station+1,n_station+1))
        out_map[i]=np.zeros((n_station+1,n_station+1))
  
    for _,row in df.iterrows():
        entry_ticks = row['entry_ticks']
        depart_ticks = row['depart_ticks']
        o = int(row['ori_id'])
        d = int(row['dest_id'])
        i = int((entry_ticks - start_tick)/slice_len)
        j = int((depart_ticks - start_tick)/slice_len)
        in_map[i][n_station][o]+=1
        out_map[j][d][n_station]+=1 

    print in_map[92][23][30]
    return in_map,out_map

def write_inout_map(in_map,out_map,o,d,To,Td,slice_len=300, start_tick=21600, end_tick=86400, n_station=104):
    o = int(o)
    d = int(d)
    n_slice = int((end_tick-start_tick)/slice_len)
    
    i = int((To-start_tick)/slice_len)
    # case 1: To and Td in the same slice
    # case 2: To is in slice i-1 and Td is in slice i
    # [Tech debt] May happen Td is in slice j>i
    if Td < start_tick+i*slice_len:
        in_map[i-1][d][o]+=1
        out_map[i-1][o][d]+=1
    else: 
        in_map[i-1][o][o]+=1
        out_map[i-1][o][o]+=1
        in_map[i][d][o]+=1
        out_map[i][o][d]+=1

    return in_map,out_map

if __name__=="__main__":
    # Step 1. Preprocessing the transaction data
    tx_filename = 'tx_data/ezlink-transaction.csv'
    msg_filename = 'tx_data/all.msg'
    #df = smu.preprocessing(tx_filename,msg_filename)
    df = smu.load_df(msg_filename)    
    
    
    o = "23" 
    d = "30"

    df = smu.get_df_by_od(df,o,d)
    df = smu.get_df_by_date(df,"2016-05-16","2016-05-16")
    df = smu.get_df_by_time(df,"14:00:00","15:00:00",mode='depart')

    slice_len=300

    
    in_map,out_map = init_inout_map(df,slice_len)
    

    """
    #observe_time_df = target_df 
    OD_time=[]

    for (x,y) in zip(observe_time_df.entry_time,observe_time_df.depart_time):
        hx,mx,sx=x.split(":")    
        hy,my,sy=y.split(":")
        ot = int(hx)*3600+int(mx)*60+int(sx)
        et = int(hy)*3600+int(my)*60+int(sy)
        if int(hx)<8 or (int(hx)==8 and int(mx)>30) or int(hx)>=9:
            continue

        if ot<=et:
            OD_time.append((ot,et))
    
    # plot the entry-depart time plots
    plt.figure(figsize=(10,5))

    for x in OD_time:
        plt.plot([x[0],x[0]],[x[0],x[1]],'-k')
    plt.savefig('test.png')
    """
 
    V = (df.duration).as_matrix()
    #V = (observe_time_df.depart_ts).as_matrix()
    X = np.transpose(np.array([V]))
    
    best_gmm, best_cv = gmm_selection(X,5)
    print "=== Derived GMM ==="
    for i,(mean,cov) in enumerate(zip(best_gmm.means_,best_gmm.covariances_)):
        print "model #"+str(i)+":"
        print "mean:",mean," cov:",cov

    clusters =  gmm_clustering(X,best_cv,best_gmm.n_components)


    """
    # Plot GMM
    x = np.array([np.linspace(min(X), max(X), 100000)]).T
    logprob = best_gmm.score_samples(x)
    pdf = np.exp(logprob)
    plt.figure(figsize=(10,5))
    plt.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
    plt.plot(x, pdf, '-k')
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    plt.savefig("["+target_od+"]"+str(observ_hour)+'-'+str(observ_min_start)+"-"+str(observ_min_end)+'.png')
    """

    member_dict=dict()
    duration_dict=dict()
    #### get members of each cluster #### 
    for i in range(len(V)):
        cid = clusters[i]
        if not cid in duration_dict:
            member_dict[cid]=[]
            duration_dict[cid]=[]
        member_dict[cid].append(i)
        duration_dict[cid].append(V[i])

    # filter out small clusters and find lucky guys for every clusters
    # default threshold: 0.10
    threshold_d = 0.10
 
    # get od_ksp dict.... key:o_d --> value: [[time,[sp],dict:o_d-->time]
    od_ksp_dict = Pickle.load(open('mrt_map/od_ksp_dict.pkl'))

    od_ksp_list = od_ksp_dict[o+"_"+d]

   
    for cid in duration_dict:
        # filter the noise
        print duration_dict[cid]
        if len(duration_dict[cid])<len(V)*threshold_d:
            continue
        # get lucky guy and his duration
        lucky_duration = min(duration_dict[cid])*60
        lucky_guy = duration_dict[cid].index(min(duration_dict[cid]))
        # find the most fit od path for lucky guy
        most_fit_i = 0
        for i in range(len(od_ksp_list)):
            cost = od_ksp_list[i][0]
            if cost>lucky_duration:
                most_fit_i=i-1
                break
        
        most_fit_cost = od_ksp_list[most_fit_i][0]
        most_fit_path = od_ksp_list[most_fit_i][1]
        most_fit_dict = od_ksp_list[most_fit_i][2]

        walking_time = (lucky_duration-most_fit_cost)/2

        print cid,most_fit_path       
        # get all the members in this cluster 
        c_df = df.iloc[member_dict[cid]].copy()
        
        c_df['T2']=c_df['depart_ticks'].apply(lambda x: x-walking_time)
        c_df['T1']=c_df['T2'].apply(lambda x: x-most_fit_cost)
   
        
        """
        *** The three rules ***
        T0 : entry_time (V)
        T3 : depart_time (V)
        T2 : T3 - walking_time
        T1 : T2 - travel_time
        ***********************
        Rule 1: T2 ~ T3 --> D
        Rule 2: T1 ~ T2 --> station along most_fit_path
        Rule 3: T0 ~ T1 --> O
        """

        # Rule 1:
        for index, row in c_df.iterrows():
            # Rule 1:
            T3 = row['depart_ticks']
            T2 = row['T2']
            in_map, out_map = write_inout_map(in_map,out_map,d,d,T2,T3) 

            # Rule 3:
            T0 = row['entry_ticks']
            T1 = row['T1']
            in_map, out_map = write_inout_map(in_map,out_map,o,o,T0,T1)

            sta_o_tick = sta_d_tick = T2
            for i in range(len(most_fit_path)-1,1,-1):
                seg_o =  most_fit_path[i-1]
                seg_d =  most_fit_path[i]
                cost = most_fit_dict[seg_o+"_"+seg_d]
                sta_o_tick -= cost
                in_map, out_map = write_inout_map(in_map,out_map,seg_o,seg_d,sta_o_tick,sta_d_tick)
                sta_d_tick=sta_o_tick

    for i in range(92,105):
        plt.matshow(in_map[i],cmap='hot', interpolation='none')
        plt.savefig(str(i)+'-in'+'.png')
        plt.matshow(out_map[i],cmap='hot', interpolation='none')
        plt.savefig(str(i)+'-out'+'.png')
        plt.close()
     

            
        
 
