import os
import random
import tqdm
import argparse
import pathlib
import shutil
from pathlib import Path
from librosa.util import find_files
from tqdm import trange
import sys
def collect_speaker_ids(roots):
    return [f.path for f in os.scandir(roots) if f.is_dir()]

def construct_speaker_id_txt(dev_speakers,dev_txt_name):
    f = open(dev_txt_name, "w")
    for dev in dev_speakers:
        f.write(dev.split('/')[-1])
        f.write("\n")
    f.close()
    return


def sample_wavs_and_dump_txt(root,dev_ids, numbers, meta_data_name):
    
    wav_list = []
    count_positive = 0
    print(f"generate {numbers} sample pairs")
    for _ in trange(numbers):
        prob = random.random()
        if (prob > 0.5):
            dev_id_pair = random.sample(dev_ids, 2)

            # sample 2 wavs from different speaker
            sample1 = random.choice(find_files(os.path.join(root,dev_id_pair[0])))
            sample2 = random.choice(find_files(os.path.join(root,dev_id_pair[1])))

            label = "0"

            wav_list.append("\t".join([label, sample1, sample2]))
            
        else:
            dev_id_pair = random.sample(dev_ids, 1)
            
            # sample 2 wavs from same speaker
            sample1 = random.choice(find_files(os.path.join(root,dev_id_pair[0])))
            sample2 = random.choice(find_files(os.path.join(root,dev_id_pair[0])))

            label = "1"
            count_positive +=1

            wav_list.append("\t".join([label, sample1, sample2]))
    print("finish, then dump file ..")
    f = open(meta_data_name,"w")
    for data in wav_list:
        f.write(data+"\n")
    f.close()

    return wav_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=19941227)
    parser.add_argument('-r', '--root', default="/nlsasfs/home/ai4bharat/gramesh/speechteam/e2e_data/pilot4_karya/aug_dc_data_raw/dc_data_clean_wav")
    parser.add_argument('-l', '--lang', default="", choices=os.listdir('/nlsasfs/home/ai4bharat/gramesh/speechteam/e2e_data/pilot4_karya/aug_dc_data_raw/dc_data_clean_wav'))
    # parser.add_argument('-o', "--output_dir", default="../../librispeech/dev_data")
    # parser.add_argument('-n',  '--speaker_num', default=40)
    parser.add_argument('-p',  '--sample_pair', default=50000)
    parser.add_argument('-v', '--split',default='valid')
    parser.add_argument('-t', '--type',default='clean')
    args = parser.parse_args()
    try:
        assert args.lang in os.listdir(args.root)
    except:
        print("select lang from", os.listdir(args.root))
        sys.exit()

    random.seed(args.seed)
    if args.type == 'clean':
        sp_root = args.root+'/'+args.lang+'/'+args.split+'/audio/'
        speakers = collect_speaker_ids(sp_root)
        sp_path = f"./downstream/indic_asv/meta_data/{args.lang}/{args.split}_speaker_ids.txt"
        dp_path = f"./downstream/indic_asv/meta_data/{args.lang}/{args.split}_data.txt"
        os.makedirs(os.path.split(sp_path)[0], exist_ok=True)
        os.makedirs(os.path.split(dp_path)[0], exist_ok=True)
        construct_speaker_id_txt(speakers, sp_path)
        wav_list = sample_wavs_and_dump_txt(sp_root, speakers, args.sample_pair, dp_path)
    elif args.type == 'noisy':
        sp_root = args.root+'/'+args.lang+'/'+args.split+'/audio/'
        speakers = collect_speaker_ids(sp_root)
        sp_path = f"./downstream/indic_asv/meta_data/{args.lang}/{args.split}_noisy_speaker_ids.txt"
        dp_path = f"./downstream/indic_asv/meta_data/{args.lang}/{args.split}_noisy_data.txt"
        os.makedirs(os.path.split(sp_path)[0], exist_ok=True)
        os.makedirs(os.path.split(dp_path)[0], exist_ok=True)
        construct_speaker_id_txt(speakers, sp_path)
        wav_list = sample_wavs_and_dump_txt(sp_root, speakers, args.sample_pair, dp_path)
    else:
        print('Not implemented')
    