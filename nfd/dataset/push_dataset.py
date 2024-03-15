import torch
# if torch.multiprocessing._actual_context is None:
# torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_descriptor')

from torch.utils.data import Dataset
import torch.nn as nn

# from torchvision.transforms import ToTensor
# import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import pickle as pkl
import os
import glob
import numpy as np

from tqdm import tqdm

import gc
from multiprocessing.pool import ThreadPool as Pool
# from multiprocessing.pool import Pool
from functools import partial
import multiprocessing as mp

import gtsam
import time
# from functools import 

# class ThresholdTransform(object):
#     def __init__(self, min_value, max_value):
#       self.min_value = 
#       pass
#     def __call__(self,x):
#       pass

def get_state(color):
    states = color[...,0].float()
    vmax = 200.
    vmin = 70.
    states = torch.clamp(states, min = vmin, max=vmax)
    states = (states-vmin)/(vmax-vmin)
    return states

def get_action(color):
    vmax = 200.
    vmin = 100.
    actions = color[...,2].float()
    actions = torch.clamp(actions, min = vmin, max=vmax)
    actions = (actions-vmin)/(vmax-vmin)
    # actions = (actions>128).to(actions.dtype)

    return actions

def get_dense_action(data):
    action_data = data['action']
    pos = np.array([p['pos'][0:2] for p in action_data])
    rot = np.array([gtsam.Rot3(*p['rot']).rpy()[0] for p in action_data]).reshape(-1,1)
    # speed = [p['speed'] for p in action_data[1:]]
    # speed.append(speed[-1])
    # speed = np.array(speed).reshape(-1,1)
    # dense_actions = torch.tensor(np.concatenate([pos,rot,speed],axis=1)).float()
    dense_actions = torch.tensor(np.concatenate([pos,rot, np.zeros_like(rot)],axis=1)).float()
    return dense_actions

def get_pose(action_data):
    # action_data = data['action']
    pos = np.array([p['pos'] for p in action_data])
    rot = np.array([p['rot'] for p in action_data])
    poses = torch.tensor(np.concatenate([rot,pos],axis=1)).float()
    return poses

def get_goal(color):
    goal = color[...,1].float()
    return goal

def sample_idx(original, proposed, perturb=True):
    if original == proposed:
        return list(range(proposed))
    elif original>proposed:
        num, div = original-1, proposed-1
        sep_list = ([num // div + (1 if x < num % div else 0)  for x in range (div)])
        
        if perturb:
            #np.random.shuffle(sep_list)
            
            perturb_num = len(sep_list)//2
            sep_list = np.array(sep_list)
            idx1 = np.random.choice(len(sep_list), size=perturb_num, replace=False)
            # print(sep_list, idx1)
            sep_list[idx1]+=1
            # print(sep_list)

            idx2 = np.random.choice(len(sep_list), size=perturb_num, replace=False)
            # print(sep_list, idx2)
            sep_list[idx2]-=1
            # print(sep_list)
            sep_list = sep_list.tolist()
            # input()
        
        index_list = [0] + sep_list
        for i in range(1, len(index_list)):
            index_list[i] = index_list[i]+index_list[i-1]
        return index_list
    elif original<proposed:
        num, div = proposed, original
        sep_list = ([num // div + (1 if x < num % div else 0)  for x in range (div)])
        if perturb:
            np.random.shuffle(sep_list)
        index_list = []
        for i, num in enumerate(sep_list):
            index_list += [i]*num
        
        return index_list
        


def test_sample_idx():
    test = (5, 5)
    print(test, sample_idx(*test))
    test = (11, 5)
    print(test, sample_idx(*test))

    test = (12, 5)
    print(test, sample_idx(*test))
    test = (13, 5)
    print(test, sample_idx(*test))

    test = (5, 11)
    print(test, sample_idx(*test))
    test = (4, 11)
    print(test, sample_idx(*test))
        

class PushDataset(Dataset):
    def __init__(self, data_dir=None, mode = 'test', index_max = None, name_filter=None, seq_length = None, shuffle=False, dense_action=False, cache=False, down_sample=5, overfit=False):
        name = 'sweeping-piles-'+mode
        if data_dir is None:
            data_dir = './data'
        if type(data_dir) is type('text'):
            self.file_list = glob.glob(os.path.join(data_dir, name, '*.pkl'))
            self.name_list = [x.split(os.sep)[-1] for x in self.file_list]
        else:
            self.file_list = []
            self.name_list = []
            # data_dir = list(data_dir)
            for dd in data_dir:
                file_list = glob.glob(os.path.join(dd,name,'*.pkl'))
                name_list = [x.split(os.sep)[-1] for x in file_list]
                self.file_list+=file_list
                self.name_list+=name_list
                # for i in range(len(file_list)):
                #     if name_list[i] not in self.name_list:
                #         self.file_list.append(file_list[i])
                #         self.name_list.append(name_list[i])
                
        self.file_list.sort(key= lambda x: x.split(os.sep)[-1])
        self.overfit = overfit
        # if mode=='test' and index_max:
        #     self.file_list = self.file_list[:index_max]
        if index_max:
            self.file_list = self.file_list[:index_max]
            # self.name_list = 
        if name_filter is not None:
            new_file_list = []
            for name in self.file_list:
                if name_filter(name):
                    new_file_list.append(name)
            self.file_list=new_file_list
        if shuffle:
            new_idx = torch.randperm(len(self.file_list)).numpy().tolist()
            self.file_list = [self.file_list[i] for i in new_idx]
        
        self.index_max = index_max
        self.seq_length = seq_length
        self.mode = mode
        self.perturb = False if mode in ['test', 'val'] else True

        # self.state_transform = 
        # self.action_transform = 
        if down_sample != 1:
            self.transform = nn.AvgPool2d(down_sample, stride=down_sample)
        else:
            self.transform = None

        # self.length_list = []
        # for fname in self.file_list:
        #   with open(fname,"rb") as f:
        #     data = pkl.load(f)
        #     self.length_list.append(len(data['color']))
        self.dense_action = dense_action
        # self.data_list = []

        self.cache = cache
        # print(f'Dataset {mode}: length {len(self.file_list)}')
        if cache: 
            # manager = mp.Manager()
            # self.data_list = manager.list()
            # self.load_all_data()
            self.data_list = [{}] *len(self.file_list)
            # self.data_list = []
    
    def load_all_data(self):    
        t0 = time.time()
        with Pool(16) as p:
            self.data_list = p.map(partial(process, transform=self.transform, file_list=self.file_list), list(range(len(self))))
        gc.collect()
        self.transform=None
        t1 = time.time()
        print(f'Dataset {self.mode}: cache loaded, time spent: {t1-t0:.2f}s')
    
    
    def load_data(self, idx):
        with open(self.file_list[idx],"rb") as f:
            try:
                data = pkl.load(f)
            except:
                print(self.file_list[idx])
                raise
            return data
            

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # if self.mode=='train' and self.index_max:
        #     idx = idx%self.index_max
        transform = self.transform
        if self.cache and self.data_list[idx]:#and len(self.data_list)==len(self):
            data = self.data_list[idx]
            states = data['states']
            actions = data['action_img']
        else:
            data = load_data(self.file_list[idx])
            color = torch.tensor(np.array(data['color']))
        
            states = get_state(color)        
            actions = get_action(color)

            if transform:
                states = transform(states)
                actions = transform(actions)

            if self.cache:
                data.pop('color', None)
                data['states'] = states
                data['action_img'] = actions
            
                # self.data_list.append(data)
                self.data_list[idx] = data


        if self.dense_action:
            dense_actions = get_dense_action(data)
  
        

        poses = get_pose(data['action'])
        

        if self.seq_length is not None:
            idx_list = sample_idx(states.shape[0], self.seq_length, self.perturb)
            states = states[idx_list,...]
            actions = actions[idx_list,...]
            poses = poses[idx_list,...]
            if self.dense_action:
                dense_actions = dense_actions[idx_list,...]

        # if transform:
            # states = transform(states)
            # actions = transform(actions)

        output = {"state":states, "action":actions, "pose": poses}

        if self.dense_action:
            #return {"state":states, "action":actions, "dense_action":dense_actions}
            output["dense_action"] = dense_actions

        # return {"state":(states-0.05)/0.5, "action":(actions-0.01)/0.1}
        return output

def process(idx, transform, file_list):
        data = load_data(file_list[idx])
        
        color = torch.tensor(np.array(data['color']))
        # print(color.dtype)
        states = get_state(color)        
        actions = get_action(color)

        if transform:
            states = transform(states)
            actions = transform(actions)
        data.pop('color', None)
        data['states'] = states
        data['action_img'] = actions
        gc.collect()
        return data

def load_data(fname):
    with open(fname,"rb") as f:
        try:
            data = pkl.load(f)
        except:
            print(fname)
            raise
        return data

def test_dataset():
    # filter = val_10k
    training_set = PushDataset(mode='val',index_max = None, name_filter=val_10k, shuffle=False, seq_length = 20, data_dir='./data', cache=True)
    dataloader = DataLoader(training_set, batch_size=16, shuffle=True, num_workers=12, pin_memory=False, persistent_workers=True)

    t1 = time.time()
    for data in dataloader:
        pass
    t2 = time.time()
    print(training_set.data_list[10]["action_img"].shape)
    print(t2-t1)
    for data in dataloader:
        pass
    t3 = time.time()
    print(t3-t2)

    
    pass

class MemoryDataset(Dataset):
    def __init__(self, dataloader):
        super().__init__()
        
        
        self.data=None
        print(f'loading data: {dataloader.dataset.mode}')

        for data in tqdm(dataloader):
            if self.data is None:
                self.data = data
                continue
            for key in data:
                # print(self.data.shape[0])
                self.data[key] = torch.cat([self.data[key], data[key]], axis=0)
        # self.length = len(dataloader.dataset)
        self.length = self.data[list(self.data.keys())[0]].shape[0]
        
        print(f'Data loaded: {dataloader.dataset.mode} Length: {self.length}')
        
    def __len__(self):
        # return self.data[self.data.keys()[0]].shape[0]
        return self.length
    
    def __getitem__(self, idx):
        return {key: self.data[key][idx,...] for key in self.data}

def MemoryDataloader(*args, **kwargs):
    dataloader = DataLoader(*args, **kwargs)
    # print(dataloader)
    new_dataset = MemoryDataset(dataloader)
    new_dataloader = DataLoader(new_dataset, *args[1:], **kwargs)
    # print(new_dataloader)
    del dataloader
    return new_dataloader


def test_memory_dataset():
    training_set = PushDataset(mode='val',index_max = None, name_filter=val_10k, shuffle=False, seq_length = 20, data_dir='./data', cache=False)
    dataloader = MemoryDataloader(training_set, batch_size=16, shuffle=True, num_workers=12, pin_memory=False, persistent_workers=True)

    # dataloader = MemoryDataloader(dataloader)
    next(iter(dataloader))
    
if __name__=="__main__":
    from nfd.dataset.data_filters import *
    # test_sample_idx()
    test_dataset()
    # test_memory_dataset()