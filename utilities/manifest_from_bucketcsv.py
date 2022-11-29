import glob,re,tqdm,os,sys
import pandas as pd
import shutil

pth = sys.argv[1] # 'kb_data_clean_wav' # set to 'kb_data_noisy_wav' for noisy split

#get all bucket_csvs

csv_list = glob.glob(f"{pth}/**/*.csv",recursive=True)
for csv in tqdm.tqdm(csv_list):
    #make the folde path 
    manifest = csv.replace('bucket.csv','manifest')
    split = manifest.split('/')[-2]

    os.makedirs(manifest,exist_ok=True) # makes the manifest directory if it does

    #read the csv
    df = pd.read_csv(csv)
    common_prefix = os.path.commonprefix(df.file_path.to_list())
    common_prefix = os.path.split(common_prefix)[0]+'/'
    
    with open(manifest+'/'+split+".tsv",'w') as tsv, \
        open(manifest+'/'+split+".wrd","w") as wrd, \
        open(manifest+'/'+split+".ltr",'w') as ltr:
        print(common_prefix,file=tsv)
        for e,name,tra,dur in df.itertuples():
            n = name.replace(common_prefix,'')
            print(n,dur,sep='\t',file=tsv)
            print(tra,file=wrd)
            print(" ".join(list(tra.replace(" ", "|"))) + " |", file=ltr)


if 'clean' in pth:
    # copy valid to train
    valids = glob.glob(f"{pth}/**/valid/**/valid.*",recursive=True)
    for v in valids:
        shutil.copy(v,v.replace('/valid/','/train/'))