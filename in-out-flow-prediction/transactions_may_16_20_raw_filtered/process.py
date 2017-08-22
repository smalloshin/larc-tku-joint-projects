import pandas as pd
import numpy as np
from sklearn import mixture
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')


from matplotlib import pyplot as plt


# decide how many gaussian to fit data
def gmm_selection(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)

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

def preprocessing(filename):
    # load the
    df = pd.read_csv(filename,dtype={"card_id":"category","ori_id":"category","dest_id":"category"})

    # concate o-d
    df['od']=df[['ori_id','dest_id']].apply(lambda x: '_'.join(x), axis=1)
    df['entry_dt']=df[['entry_date','entry_time']].apply(lambda x: ' '.join(x), axis=1)
    # compute entry_ts
    df['entry_ts']=df['entry_dt'].apply(lambda x: time.mktime((datetime.strptime(x,"%Y-%m-%d %H:%M:%S")).timetuple()))
    # recover all the information of departure
    df['depart_ts']=df.entry_ts+df.duration*60
    df['depart_date']=df['depart_ts'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
    df['depart_time']=df['depart_ts'].apply(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S"))
    df.drop('entry_dt',axis=1,inplace=True)

    df.to_msgpack("all.msg")
    return df


if __name__=="__main__":
    """
    filename = 'ezlink-transaction.csv'
    df = preprocessing(filename)
    """ 
    
    # load df for ezlink transaction data
    df = pd.read_msgpack("all.msg")

    # groupsize=df.groupby(['od']).size()
    # print groupsize.sort_values(ascending=False).head(10)

    o = "24"
    d = "27"
    target_od=o+"_"+d
    #target_od_1="67_27"

    target_df = df[df.od==target_od].copy()
    target_mean = target_df.mean()
    target_std = target_df.std()

    """
    print "******************************"
    print "=== head of dataframe ==="
    print target_df.head()
    print "=== mean ==="
    print target_mean
    print "=== std ==="
    print target_std
    print "******************************"
    """   

    # remove outlier 
    target_df['dmin'] = target_df['duration'].astype(int)
    s = float(target_df['dmin'].value_counts().sum())
    target_counts =  target_df['dmin'].value_counts().values/s
    print "Targeted Dataframe Shape:",target_df.shape

    threshold = 0.01
    valid_minutes = []

    i = -1
    for x in target_counts:
         if x>=threshold:
             i=i+1
         else:
             break

    valid_minutes =  target_df['dmin'].value_counts().index[0:i] 
    max_minutes =  max(valid_minutes)
    min_minutes =  min(valid_minutes)
    clean_df = target_df[(target_df.duration>=min_minutes) & (target_df.duration<max_minutes+1) ]
    print "******************************"
    print "Cleaned DataFrame Shape:",clean_df.shape  
    print "******************************"

    observ_hour = 8
    observ_min_start=0
    observ_min_end=5

    target_df['depart_hour']=target_df['depart_time'].apply(lambda x: x.split(":")[0]).astype(int)
    target_df['depart_min']=target_df['depart_time'].apply(lambda x: x.split(":")[1]).astype(int)
    #observe_time_df = target_df[(target_df.entry_date=="2016-05-16") & (target_df.depart_hour==observ_hour) & (target_df.depart_min>=observ_min_start) & (target_df.depart_min<observ_min_end)]
  
    observe_time_df = target_df[(target_df.depart_hour==observ_hour) & (target_df.depart_min>=observ_min_start) & (target_df.depart_min<observ_min_end)]

    print "******************************"
    print "Observed DataFrame Shape:",observe_time_df.shape
    print "******************************"



    """ 
    print observe_time_df.describe()
    print datetime.fromtimestamp(min(observe_time_df.entry_ts))
    print datetime.fromtimestamp(max(observe_time_df.entry_ts))   
    """
    X = np.transpose(np.array([(observe_time_df.duration).as_matrix()]))
    
    best_gmm, best_cv = gmm_selection(X)
    print "=== Derived GMM ==="

    # Plot GMM
    x = np.array([np.linspace(min(X), max(X), 100000)]).T
    logprob = best_gmm.score_samples(x)
    pdf = np.exp(logprob)

    """
    fig = plt.figure(figsize=(5, 1.7))
    fig.subplots_adjust(left=0.12, right=0.97,
                    bottom=0.21, top=0.9, wspace=0.5)

    ax = fig.add_subplot(151)
    ax.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
    ax.plot(x, pdf, '-k')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')
    fig.savefig('test.png')
    """
    plt.figure(figsize=(10,5))
    plt.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
    plt.plot(x, pdf, '-k')
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    plt.savefig("["+target_od+"]"+str(observ_hour)+'-'+str(observ_min_start)+"-"+str(observ_min_end)+'.png')

    for i,(mean,cov) in enumerate(zip(best_gmm.means_,best_gmm.covariances_)):
        print "model #"+str(i)+":"
        print "mean:",mean," cov:",cov

    #### BACKFORWARD ####   
    print gmm_clustering(X,best_cv,best_gmm.n_components)

     
