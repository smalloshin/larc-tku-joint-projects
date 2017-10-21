import pandas as pd

def parse_ym(x):
    s = x.split(" ")
    if len(s)==0:
        return "",""
    elif len(s)==2:
        return s[0],s[1]
    else:
        return "",x

if __name__=="__main__":
    columns = ['uid','start','end','title','location','company_name']
    df = pd.read_csv('20160111.profile.csv',header = None, names = columns, dtype="category")
    df.start = df.start.astype(str)
    df.end = df.end.astype(str)
    
    print df.describe()
    df['start_m']=df['start'].apply(lambda x: parse_ym(x)[0])
    df['start_y']=df['start'].apply(lambda x: parse_ym(x)[1])
    df['end_m']=df['end'].apply(lambda x: parse_ym(x)[0])
    df['end_y']=df['end'].apply(lambda x: parse_ym(x)[1])
 
    # clean the
    both_empty = df[(df.end_y=="nan") & (df.start_y=="nan")].index
    df.drop(both_empty,inplace=True)
    print df.describe()


    print df.head()
    
