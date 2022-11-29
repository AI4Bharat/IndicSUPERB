import os, glob, pandas as pd, tqdm
import soundfile as sf
import string
import re,os, sys
from joblib import Parallel, delayed
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import shutil

#Get transcripts from all subsets of data

lang2lcode = {
    'bengali':'bn',
    'gujarati':'gu',
    'hindi':'hi',
    'kannada':'kn',
    'malayalam':'ml',
    'marathi':'mr',
    'odia':'or',
    'punjabi':'pa',
    'sanskrit':'sa',
    'tamil':'ta',
    'telugu':'te',
    'urdu':'ur'
}

def read_file_length(path):
    return len(sf.read(path)[0])

def normalize(text, language):
    lang_code = lang2lcode[language]
    text = text.translate(str.maketrans('', '', string.punctuation+'।'))
    text = re.sub(r' ?(\d+) ?', r' \1 ', text)
    if lang_code in ['ur']:
        return text
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang_code)
    return normalizer.normalize(text)

def process(transcript_relative):
    transcript = os.path.abspath(transcript_relative)
    lang = transcript.split('/')[-3]
    #read a transcript and prepare a bucket.csv file containing <path> <duration> <transcript>
    df = pd.read_csv(transcript,header=None,sep='\t',names=['file_path','transcript'])
    df['file_path'] = df['file_path'].apply(lambda x: transcript.replace('transcription.txt','audio/')+x.split('-')[1]+'/'+x.replace('.m4a','.wav'))
    df['length'] = df['file_path'].apply(lambda x: read_file_length(x))
    df['transcript'] = df['transcript'].apply(lambda x: normalize(x.strip().strip('॥').strip('৷').strip('।').strip(),lang))
    df.to_csv(transcript.replace('transcription.txt','bucket.csv'),index=False)
    # print(transcript)
    print("Done",transcript,'\n')


path = sys.argv[1] # 'kb_data_clean_wav'
all_transcripts = glob.glob(f'{path}/**/transcription.txt',recursive=True)
Parallel(n_jobs=-120)(
    delayed(process)(t) for t in tqdm.tqdm(all_transcripts)
)

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
    