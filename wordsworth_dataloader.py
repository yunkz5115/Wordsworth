# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:08:31 2024

@author: yunkz
"""
import pathlib #读图片路径
import numpy as np
import torch
from scipy.io.wavfile import read
from torchaudio import transforms
from tqdm import trange
import os
from shutil import copyfile
import zipfile

class WaveDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_label_dir, transform=transforms.Resample(orig_freq=24000, new_freq=8000)):
        super().__init__()
        
        all_images_paths = list(img_dir.glob('*/*')) #Get all file pathway

        all_images_paths = [str(path) for path in all_images_paths] 
        
        #random.shuffle(all_images_paths) #Shuffle
        
        image_count = len(all_images_paths) #visualize length of filelist
    
        label_names = sorted(item.name for item in img_dir.glob('*/') if item.is_dir())

        label_to_index = dict((name,index)for index,name in enumerate(label_names)) #Convert file name to number label
    
        all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
        
        self.img_dir = all_images_paths
        #self.img_labels = pd.read_csv(img_label_dir)  # Optional dataframe with 0 name and 1 type
        self.img_labels = all_images_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)  # Get dataset length
    
    def __num_label__(self):
        return np.unique(self.img_labels).shape[0]  # Get dataset label length
    
    def __getitem__(self, index):
        # Get image file folder
        img_path = self.img_dir[index]
        image = read(img_path)
        image = image[1]
        if len(image)>36000:
            image = np.delete(image, range(36000,len(image)))
        elif len(image)<36000:
            image = np.hstack([image,np.zeros(36000-len(image))])
        #print(img_path)
        image = torch.from_numpy(image.reshape([1,image.shape[0]])).type(torch.FloatTensor)  # tensor type
        label = self.img_labels[index]
        if self.transform is not None:
            image = self.transform(image)  # apply transform
        
        return image, label

class CochleagramDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_label_dir, transform=transforms.Resize([128,128])):
        super().__init__()
        
        all_images_paths = list(img_dir.glob('*/*')) # #Get all file pathway

        all_images_paths = [str(path) for path in all_images_paths]
        
        #random.shuffle(all_images_paths) #Shuffle
        
        image_count = len(all_images_paths) #visualize length of filelist
    
        label_names = sorted(item.name for item in img_dir.glob('*/') if item.is_dir())

        label_to_index = dict((name,index)for index,name in enumerate(label_names)) #Convert file name to number label
    
        all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
        
        self.img_dir = all_images_paths
        #self.img_labels = pd.read_csv(img_label_dir)  # Optional dataframe with 0 name and 1 type
        self.img_labels = all_images_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)  # Get dataset length
    
    def __num_label__(self):
        return np.unique(self.img_labels).shape[0]  # Get dataset label length
    
    def __getitem__(self, index):
        # Get image file folder
        img_path = self.img_dir[index]
        image = np.loadtxt(img_path)
        #We added 12 samples padding for cochleagrams
        image = np.hstack([np.zeros([211,12]),image[:,:int(np.ceil(image.shape[1]*2/3))]])
        image = torch.from_numpy(image.reshape([1,image.shape[0],image.shape[1]])).type(torch.FloatTensor)  # tensor type
        label = self.img_labels[index]
        if self.transform is not None:
            image = self.transform(image)  # apply transform
        
        return image, label
    
def using_comprehension(word, sentence):
    loc = [n for n in range(len(sentence)) if sentence.find(word, n) == n]
    return loc

class WaveDataLoader_Voice(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_label_dir, transform=transforms.Resample(orig_freq=24000, new_freq=8000)):
        super().__init__()
        
        all_images_paths = list(img_dir.glob('*/*')) #Get all file pathway

        all_images_paths = [str(path) for path in all_images_paths] 
        
        #random.shuffle(all_images_paths) #Shuffle
        
        image_count = len(all_images_paths) #visualize length of filelist
    
        label_names = sorted(item.name for item in img_dir.glob('*/') if item.is_dir())

        label_to_index = dict((name,index)for index,name in enumerate(label_names)) #Convert file name to number label
    
        all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
        
        self.img_dir = all_images_paths
        #self.img_labels = pd.read_csv(img_label_dir)  # Optional dataframe with 0 name and 1 type
        self.img_labels = all_images_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)  # Get dataset length
    
    def __num_label__(self):
        return np.unique(self.img_labels).shape[0]  # Get dataset label length
    
    def __getitem__(self, index):
        # Get image file folder
        img_path = self.img_dir[index]
        loc = using_comprehension('_', img_path)
        voice_type = img_path[(loc[-2]+1):loc[-1]]
        image = read(img_path)
        image = image[1]
        if len(image)>36000:
            image = np.delete(image, range(36000,len(image)))
        elif len(image)<36000:
            image = np.hstack([image,np.zeros(36000-len(image))])
        #print(img_path)
        image = torch.from_numpy(image.reshape([1,image.shape[0]])).type(torch.FloatTensor)  # tensor type
        label = self.img_labels[index]
        if self.transform is not None:
            image = self.transform(image)  # apply transform
        
        return image, label, voice_type

def generate_WW_subset(root_path,target_path,word,speaking_rate='_',talker='_',accent='_',model='_',attributes='copy'):
    filelist=os.listdir(root_path+'train/'+word+'/')
    if attributes=='copy':
        for token in trange(filelist):
            if (word in token)&(speaking_rate in token)&(talker in token)&(accent in token)&(model in token):
                copyfile(root_path+'train/'+word+'/'+token, target_path+'/'+word+'/'+token)
    elif attributes=='zip':
        zfile=zipfile.ZipFile(target_path+'/'+'WW_subset.zip',"w")
        for token in trange(filelist):
            if (word in token)&(speaking_rate in token)&(talker in token)&(accent in token)&(model in token):
                zfile.write(root_path+'train/'+word+'/'+token)
        zfile.close()
