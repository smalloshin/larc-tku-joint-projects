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


if __name__=="__main__":
    # Step 1. Preprocessing the transaction data
    tx_filename = 'tx_data/ezlink-transaction.csv'
    msg_filename = 'tx_data/all.msg'
    df = smu.preprocessing(tx_filename,msg_filename)
    print df.head()
    exit()

    
    o = "24"
    d = "27"
    target_od=o+"_"+d
    #target_od_1="67_27"

    target_df = df[df.od==target_od].copy()
    target_mean = target_df.mean()
    target_std = target_df.std()
    

    observ_hour = 8
    observ_min_start=0
    observ_min_end=5

    target_df['depart_hour']=target_df['depart_time'].apply(lambda x: x.split(":")[0]).astype(int)
    target_df['depart_min']=target_df['depart_time'].apply(lambda x: x.split(":")[1]).astype(int)
    #observe_time_df = target_df[(target_df.entry_date=="2016-05-16") & (target_df.depart_hour==observ_hour) & (target_df.depart_min>=observ_min_start) & (target_df.depart_min<observ_min_end)]
  
    
    #observe_time_df = target_df[(target_df.depart_hour==observ_hour) & (target_df.depart_min>=observ_min_start) & (target_df.depart_min<observ_min_end)]

    observe_time_df = target_df[(target_df.entry_date=="2016-05-17")]
    print "******************************"
    print "Observed DataFrame Shape:",observe_time_df.shape
    print "******************************"

    

    """ 
    print observe_time_df.describe()
    print datetime.fromtimestamp(min(observe_time_df.entry_ts))
    print datetime.fromtimestamp(max(observe_time_df.entry_ts))   
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
     
 
    OD_time = sorted(OD_time, key=lambda x:x[1])
    V = (observe_time_df.duration).as_matrix()
    #V = (observe_time_df.depart_ts).as_matrix()

    X = np.transpose(np.array([V]))
    
    best_gmm, best_cv = gmm_selection(X,5)
    print "=== Derived GMM ==="

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

    for i,(mean,cov) in enumerate(zip(best_gmm.means_,best_gmm.covariances_)):
        print "model #"+str(i)+":"
        print "mean:",mean," cov:",cov

    clusters =  gmm_clustering(X,best_cv,best_gmm.n_components)

    #### get members of each cluster #### 
    cluster_dict = dict()
    for i in range(len(V)):
        cid = clusters[i]
        if not cid in cluster_dict:
            cluster_dict[cid]=[]
        cluster_dict[cid].append(V[i])

    # default threshold: 0.10
    threshold_d = 0.10

    for cid in cluster_dict:
        # filter the noise
        if len(cluster_dict[cid])<len(V)*threshold_d:
            continue
        # fit the path
        lucky_guy = min(cluster_dict[cid])
        print cid,lucky_guy
    
   
