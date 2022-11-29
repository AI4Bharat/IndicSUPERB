import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import re
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
from pathlib import Path

# CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


# Indic SUPERB
class SpeakerClassifiDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, max_timestep=None):

        self.root = file_path
        
        self.max_timestep = max_timestep
        print(file_path)
        self.usage_list = glob.glob(f'{file_path}/**/*.wav',recursive=True)
        self.classes = sorted(list(set(os.path.split(x)[1].split('-')[1] for x in self.usage_list if any(y in x for y in ['train','valid','test_known']))))
        self.class_num = len(self.classes)
        print(len(self.usage_list))
        print(self.classes)

        cache_path = os.path.join(meta_data, f'{mode}.pkl')
        if os.path.isfile(cache_path):
            print(f'[SpeakerClassifiDataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset = eval("self.{}".format(mode))()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'[SpeakerClassifiDataset] - there are {len(dataset)} files found')

        mapping_path = os.path.join(meta_data, 'mapping.pkl')
        if os.path.isfile(mapping_path):
            print(f'Loading label mappings from {mapping_path}')
            with open(mapping_path,'rb') as reader:
                self.label2class_name = pickle.load(reader)
            self.class_name2label = {k:v for v,k in self.label2class_name.items()}
            assert len(self.class_name2label.keys()) == self.class_num
        else:
            self.class_name2label = {k:v for v,k in enumerate(self.classes)}
            self.label2class_name = {k:v for v,k in self.class_name2label.items()}

            with open(mapping_path,'wb') as writer:
                pickle.dump(self.label2class_name,writer)

        print(self.class_name2label)

        self.dataset = dataset
        self.label = self.build_label(self.dataset)


    # file_path/id0001/asfsafs/xxx.wav
    def build_label(self, train_path_list):
        y = []
        for path in train_path_list:
            y.append(self.class_name2label[os.path.split(path)[1].split('-')[1]])
        return y
        
    def get_label_to_class(self):
        return self.label2class_name

    # @classmethod
    # def label2class(self, labels):
    #     # print(labels)
    #     print(self)
    #     mapping_path = os.path.join(CACHE_PATH, f'mapping.pkl')
    #     with open(mapping_path,'rb') as reader:
    #         label2class_name = pickle.load(reader)
    #     return [label2class_name[x] for x in labels]
    
    def train(self):

        dataset = []
        print("search specified wav name for training set")
        for string in tqdm.tqdm(self.usage_list):
            if 'train' in string:
                dataset.append(string)
        print("finish searching training set wav")
                
        return dataset
        
    def valid(self):

        dataset = []
        print("search specified wav name for dev set")
        for string in tqdm.tqdm(self.usage_list):
            if 'valid' in string:
                dataset.append(string)
        print("finish searching dev set wav")

        return dataset       

    def test_known(self):

        dataset = []
        print("search specified wav name for test set")
        for string in tqdm.tqdm(self.usage_list):
            if 'test_known' in string:
                dataset.append(string)
        print("finish searching test set wav")

        return dataset

    def test_known_noisy(self):
        fp = str(self.root).replace('dc_data_clean_wav','dc_data_noisy_wav')
        print(fp)
        usage_list = glob.glob(f"{fp}/**/*.wav",recursive=True)
        dataset = []
        print("search specified wav name for test known noisy set")
        for string in tqdm.tqdm(usage_list):
            if 'test_known' in string:
                dataset.append(string)
        print("finish searching test unknown set wav")

        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep

        # def path2name(path):
        #     return Path("-".join((Path(path).parts)[-3:])).stem

        path = self.dataset[idx]
        return wav.numpy(), self.label[idx], path
        
    def collate_fn(self, samples):
        return zip(*samples)
