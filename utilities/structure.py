# Given a language, take all splits inside and make a folder structure
'''
    filename = uniqueid-sid-gender.m4a
     lang
       |- train
            |- speaker1 
            |- speaker2 
'''
# Exploit use of their homogeneous structure

import glob,shutil, tqdm, os
from joblib import Parallel, delayed
import subprocess,sys

folder_path = sys.argv[1] # 'kb_data_clean_m4a'
output_folder = sys.argv[2] # 'kb_data_clean_wav'
language=sys.argv[3]

all_files = glob.glob(f"{folder_path}/{language}/**/*.m4a",recursive=True)

def process_audio(input_file, output_file):
    subprocess.call(['ffmpeg', '-i', input_file,'-ar', '16k', '-ac', '1', '-hide_banner', '-loglevel', 'error', output_file]) 

def structure_file(filepath):
    folder, filename = os.path.split(filepath)
    
    folder = folder.replace(folder_path,output_folder) +'/'+ filename.split('-')[1] #Create a proper path
    os.makedirs(folder,exist_ok=True)
    process_audio(filepath,folder+'/'+filename.replace('.m4a','.wav'))
    #shutil.copy(filepath,folder+'/'+filename)

def structure_file_sid(filepath):
    folder, filename = os.path.split(filepath)
    
    folder = folder.replace(folder_path,output_folder) #Create a proper path
    os.makedirs(folder,exist_ok=True)
    process_audio(filepath,folder+'/'+filename.replace('.m4a','.wav'))
    #shutil.copy(filepath,folder+'/'+filename)

def copy_transcripts():
    all_transcripts = glob.glob(f"{folder_path}/**/*.txt",recursive=True)
    print('Copying Transcripts')
    for t in tqdm.tqdm(all_transcripts):
        folder, filename = os.path.split(t)
        folder = folder.replace(folder_path,output_folder)  #Create a proper path
        os.makedirs(folder,exist_ok=True)
        shutil.copy(t,folder+'/'+filename)

copy_transcripts()

Parallel(n_jobs=-16, backend='multiprocessing')(
  delayed(structure_file)(fp) for fp in tqdm.tqdm(all_files)
)