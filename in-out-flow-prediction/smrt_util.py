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
import datetime as dt
import  _pickle as Pickle 
# import cPickle as Pickle

must_fields = [u'card_id', u'ori_id', u'dest_id', u'entry_date', u'entry_time',u'duration', u'od', u'entry_ts', u'depart_ts', u'depart_date',u'depart_time']

# load file and return DataFrame
def load_df(filename):
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename,dtype={"card_id":"category","ori_id":"category","dest_id":"category"})        
    elif filename.lower().endswith('.msg'):
        df = pd.read_msgpack(filename)
    else:
        print("[ERROR] Not a valid format. Must be *.csv or *.msg")
        exit()
    # check the validity
    for field in must_fields:
        if not field in df.columns:
            print("[Warning] Column "+field+" does not exist. It is suggested to execute the function 'preprocess' first.") 
    
    return df

def processing(tx_filename, msg_filename):
    # Step 1.1: load csv
    prev_time = time.time()
    # load the
    print("[Info] Preprocessing start.... (May take long time)")
    print("*. Step 1. Load csv file")
    df = pd.read_csv(tx_filename,dtype={"card_id":"category","ori_id":"category","dest_id":"category"})
    print("Done!", time.time()-prev_time)

    prev_time = time.time()
    # concate o-d
    print("*. Step 2. Concate O-D")
    print("   - 2.1. transform internal id to node id")
    iid_dict = Pickle.load(open("mrt_map/iid2nid_dict.pkl","rb"))
    df['ori_id']=df['ori_id'].apply(lambda x: iid_dict[x])
    df['dest_id']=df['dest_id'].apply(lambda x: iid_dict[x])
    print("   --> Done!", time.time()-prev_time)

    ttt = time.time()
    print("   - 2.2. Concate o and d")
    df['od']=df[['ori_id','dest_id']].apply(lambda x: '_'.join(x), axis=1)
    print("   --> Done!", time.time()-ttt)
    print("Done!", time.time()-prev_time)

    prev_time = time.time()
    print("*. Step 3. Transfer date time into time stamp")
    print("   - 3.1. Compute entry timestamp")
    df['entry_dt']=df[['entry_date','entry_time']].apply(lambda x: ' '.join(x), axis=1)
    # compute entry_ts
    df['entry_ts']=df['entry_dt'].apply(lambda x: time.mktime((datetime.strptime(x,"%Y-%m-%d %H:%M:%S")).timetuple()))
    print("   --> ", time.time()-prev_time)

    tt = time.time()
    # recover all the information of departure
    print("   - 3.2. Compute depart timestamp")
    df['depart_ts']=df.entry_ts+df.duration*60
    df['depart_date']=df['depart_ts'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
    df['depart_time']=df['depart_ts'].apply(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S"))
    df['h']=df['entry_time'].apply(lambda x: x.split(":")[0]).astype(int)
    df['m']=df['entry_time'].apply(lambda x: x.split(":")[1]).astype(int)
    df['s']=df['entry_time'].apply(lambda x: x.split(":")[2]).astype(int)
    df['entry_ticks']=3600*df['h']+60*df['m']+df['s']

    df['h']=df['depart_time'].apply(lambda x: x.split(":")[0]).astype(int)
    df['m']=df['depart_time'].apply(lambda x: x.split(":")[1]).astype(int)
    df['s']=df['depart_time'].apply(lambda x: x.split(":")[2]).astype(int)
    df['depart_ticks']=3600*df['h']+60*df['m']+df['s']
    df.drop('entry_dt',axis=1,inplace=True)
    df.drop('h',axis=1,inplace=True)
    df.drop('m',axis=1,inplace=True)
    df.drop('s',axis=1,inplace=True)
    print("Finished:",time.time()-tt)

    prev_time = time.time()
    print("*. Step 4. Save DataFrame")
    df.to_msgpack(msg_filename)
    print("Done!", time.time()-prev_time)

def preprocessing_python2(tx_filename,msg_filename):
    prev_time = time.time()
    # load the
    print("[Info] Preprocessing start.... (May take long time)")
    print("*. Step 1. Load csv file")
    df = pd.read_csv(tx_filename,dtype={"card_id":"category","ori_id":"category","dest_id":"category"})
    print("Done!", time.time()-prev_time)
   
    prev_time = time.time()
    # concate o-d
    print("*. Step 2. Concate O-D")

    print("   - 2.1. transform internal id to node id")
    iid_dict = Pickle.load(open("mrt_map/iid2nid_dict.pkl","rb"))
    df['ori_id']=df['ori_id'].apply(lambda x: iid_dict[x])
    df['dest_id']=df['dest_id'].apply(lambda x: iid_dict[x])
    print("   --> Done!", time.time()-prev_time)
    ttt = time.time()
    print("   - 2.2. Concate o and d")
    df['od']=df[['ori_id','dest_id']].apply(lambda x: '_'.join(x), axis=1)
    print("   --> Done!", time.time()-ttt)
    print("Done!", time.time()-prev_time)

 
    prev_time = time.time()
    print("*. Step 3. Transfer date time into time stamp")
    print("   - 3.1. Compute entry timestamp")
    df['entry_dt']=df[['entry_date','entry_time']].apply(lambda x: ' '.join(x), axis=1)
    # compute entry_ts
    df['entry_ts']=df['entry_dt'].apply(lambda x: time.mktime((datetime.strptime(x,"%Y-%m-%d %H:%M:%S")).timetuple()))
    print("   --> ", time.time()-prev_time)

    tt = time.time()
    # recover all the information of departure
    print("   - 3.2. Compute depart timestamp")
    df['depart_ts']=df.entry_ts+df.duration*60
    df['depart_date']=df['depart_ts'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
    df['depart_time']=df['depart_ts'].apply(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S"))
    df['h']=df['entry_time'].apply(lambda x: x.split(":")[0]).astype(int)
    df['m']=df['entry_time'].apply(lambda x: x.split(":")[1]).astype(int)
    df['s']=df['entry_time'].apply(lambda x: x.split(":")[2]).astype(int)
    df['entry_ticks']=3600*df['h']+60*df['m']+df['s']

    df['h']=df['depart_time'].apply(lambda x: x.split(":")[0]).astype(int)
    df['m']=df['depart_time'].apply(lambda x: x.split(":")[1]).astype(int)
    df['s']=df['depart_time'].apply(lambda x: x.split(":")[2]).astype(int)
    df['depart_ticks']=3600*df['h']+60*df['m']+df['s']
    
    df.drop('entry_dt',axis=1,inplace=True)
    df.drop('h',axis=1,inplace=True)
    df.drop('m',axis=1,inplace=True)
    df.drop('s',axis=1,inplace=True)

    print("   --> ", time.time()-tt)
    print("Done!", time.time()-prev_time)

    prev_time = time.time()
    print("*. Step 4. Save DataFrame")
    df.to_msgpack(msg_filename)
    print("Done!", time.time()-prev_time)
    return df

    
def get_df_by_od(df,o,d):
    target_od=o+"_"+d
    df = df[df.od==target_od].copy()
    return df    

# date format: yyyy-mm-dd
def get_df_by_date(df,date_start,date_end,mode='entry'):
    ds = date_start+" 00:00:00"
    de = date_end+" 23:59:59"
    start_ts = time.mktime((datetime.strptime(ds,"%Y-%m-%d %H:%M:%S")).timetuple())  
    end_ts = time.mktime((datetime.strptime(de,"%Y-%m-%d %H:%M:%S")).timetuple())
    if mode=='entry':
        return df[(df.entry_ts>=start_ts) & (df.entry_ts<=end_ts)]
    elif mode=='depart':
        return df[(df.depart_ts>=start_ts) & (df.depart_ts<=end_ts)]
    else:
        print("[Error] Mode is error. Should be one of 'entry' or 'depart'")
        exit()        

# format: 12:23 (second will be ignore)
def get_df_by_time(df,time_start, time_end,mode="entry"):
    ttt = time.time()
    """
    if mode=='entry':
        df['h']=df['entry_time'].apply(lambda x: x.split(":")[0]).astype(int)
        df['m']=df['entry_time'].apply(lambda x: x.split(":")[1]).astype(int)
    elif mode=='depart':
        df['h']=df['depart_time'].apply(lambda x: x.split(":")[0]).astype(int)
        df['m']=df['depart_time'].apply(lambda x: x.split(":")[1]).astype(int)
    else:
        print "[Error] Mode is error. Should be one of 'entry' or 'depart'"
        exit()
    df['ticks']=df['h']*3600+df['m']*60
    """
    hx,mx,sx=time_start.split(":")
    hy,my,sy=time_end.split(":")
    ot = int(hx)*3600+int(mx)*60+int(sx)
    et = int(hy)*3600+int(my)*60+int(sy)

    df = df[(df[mode+"_ticks"]>=ot) & (df[mode+"_ticks"]<=et)].copy()
    return df


def ticks_to_slice(ticks,slice_len=300, start_ticks=16800, end_ticks=90000):
    slice_no = -1
    if ticks>=start_ticks and ticks<=end_ticks:
        slice_no = int((ticks - start_ticks)/slice_len)
    return slice_no

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


def find_the_best_fitting_path(df_cid,od,od_ksp_dict):
    # get lucky guy and his duration
    lucky_duration = min(df_cid.duration)*60
    lucky_guy = df_cid.duration.idxmin()
    #print(df_cid,od)
    # od_ksp_dict:
    od_ksp_list=od_ksp_dict[od]
    #print(od_ksp_list)
    
    # find the most fit od path for lucky guy
    most_fit_i = 0
    for i in range(len(od_ksp_list)):
        cost = od_ksp_list[i][0]
        if cost>lucky_duration and i>0:
            most_fit_i=i-1
            break
    
    return od_ksp_list[most_fit_i],lucky_duration

def dict_updater(pr_dict,n_dict,w_dict,o,d,entry_slice,pid,walking_time):
    """
    goal: update the dictionaries with o,d,entry_slice,pid,walking_time
    return: pr_dict, n_dict, w_dict        
    
    (note) pr_dict needs to divide n_dict to get conditional probabilities later.
           n_dict is total number of paths with entry:o and entry_slice
    """

    # initial part...
    if o not in pr_dict:
        pr_dict[o]=dict()
        n_dict[o]=dict()
        w_dict[o]=dict()
    if d not in pr_dict[o]:
        pr_dict[o][d]=dict()
        w_dict[o][d]=dict()
    if entry_slice not in pr_dict[o][d]:
        pr_dict[o][d][entry_slice]=dict()
        w_dict[o][d][entry_slice]=dict()
        n_dict[o][entry_slice]=1.
    if pid not in pr_dict[o][d][entry_slice]:
        pr_dict[o][d][entry_slice][pid]=1.
        w_dict[o][d][entry_slice][pid]=[walking_time]
    # update part
    try:
        pr_dict[o][d][entry_slice][pid]+=1.
        n_dict[o][entry_slice]+=1.
        w_dict[o][d][entry_slice][pid].append(walking_time)
    except:
        print(o,d,entry_slice,pid)
    
    return pr_dict,n_dict,w_dict

def main():
    df = preprocessing('ezlink-transaction.csv') 
    # df = load_df("all.msg")
    df = get_df_by_date(df,"2016-05-16","2016-05-18")
    df = get_df_by_od(df,"24","27")
    print(get_df_by_time(df,"11:00:00","13:00:00"))

if __name__=="__main__":
    main()

