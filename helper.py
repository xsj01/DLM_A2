import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
from datetime import datetime
from nfd.dataset import PushDataset, MemoryDataloader
from nfd.dataset.dynamics_dataset import DynamicsDataset
from cliport.environments.environment import PilesEnvironment
from nfd.agents import Base
from nfd.utils.logger import MetricTracker
import nfd.utils.utils as utils

from IPython.display import HTML
from IPython.display import display
import base64

class Trainer(Base):
    def __init__(self, cfg, model, optimizer, dataloader, criterion, metric_ftns, scheduler=None, name='test',run_id='', comments='', stage=1, epoch_offset=0):
        super().__init__(cfg)

        self.assets_root = cfg['assets_root']
        self.dtype = torch.float
        self.to(torch.device('cuda' if cfg['gpu'] else 'cpu'))

        self.loss_history={'train':[],'val':[]}
        
        self.model = model.to(self.device)
        self.optim = optimizer
        self.dataloaders = dataloader
        self.scheduler = scheduler
        # self.loss_fns = loss_fn
        self.criterion = criterion
        self.batch_size = cfg.train.batch_size

        self.auto_empty_cache = True
       
        if type(metric_ftns) is not list:
            metric_ftns = [metric_ftns]
        self.metric_ftns = metric_ftns
        for met in self.metric_ftns:
            if '__name__' not in dir(met): met.__name__ = met.__class__.__name__
        # self.metric_names = [m.__name__ for m in self.metric_ftns]

        self.num_epochs = cfg.train.epochs
        self.best_epoch = 0
        self.best_loss_val = np.inf
        self.best_model_path = None
        self.name = name
        self.run_id= run_id

        save_dir = os.path.join(cfg.train.save_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

    def to(self, device):
        self.device=device
    
    def set_seed(self, seed=None):
        if seed:
            utils.set_seed(seed)
    
    def empty_cache(self, force=False):
        if force or self.auto_empty_cache:
            torch.cuda.empty_cache(); gc.collect()
    
    def train(self):
        best_loss_val = np.inf
        metric_hist = {'loss':[], 'error_single':[], 'error_multi':[]}
        for epoch in range(self.num_epochs):

            loss = self.training_epoch()
            metric_hist['loss'].append(loss)

            val_metric = self.validation_epoch()
            metric_hist['error_single'].append(val_metric[0])
            metric_hist['error_multi'].append(val_metric[1])
            print(f'Epoch {epoch} Training Loss: {loss:.3E}, Error (Single): {val_metric[0]:.3E}, Error (Multi): {val_metric[1]:.3E}')
            
            if val_metric[-1]<best_loss_val:
                best_loss_val = val_metric[-1]
                path = os.path.join( self.save_dir, 'model_best.pt' )
                torch.save(self.model, path)
        return metric_hist
        pass

    def training_epoch(self):
        self.phase = 'train'
        self.model.train()
        dataloader = self.dataloaders[self.phase]

        tracker = MetricTracker('loss')

        for data in dataloader:
            self.optim.zero_grad()

            n_particle = data['object_pose'].shape[-2]
            bs = data['pose'].shape[0]
            
            pusher_traj = torch.cat([data['pose'][:,:-1,None,-3:-1], data['pose'][:,1:,None,-3:-1]], dim=2).to(self.device)
            pusher_traj = pusher_traj.view(-1,2,2)
            state_0_gt = data['object_pose'][:,:-1,:, -3:-1].to(self.device).view(-1,n_particle,2)
            state_1_gt = data['object_pose'][:,1:,:, -3:-1].to(self.device).view(-1,n_particle,2)

            state_1_pred = self.model(state_0_gt, pusher_traj)
            loss = self.criterion(state_1_pred,state_1_gt, state_0_gt)

            loss.backward()
            self.optim.step()
            tracker.update('loss', loss.item(), bs)

        self.empty_cache()
        return tracker.avg('loss')

    def validation_epoch(self):
        self.phase = 'val'
        dataloader = self.dataloaders[self.phase]

        tracker = MetricTracker(*[m.__name__+f'/{i}' for m in self.metric_ftns for i in ['single', 'multi']])

        self.model.eval()

        for data in dataloader:

            n_particle = data['object_pose'].shape[-2]
            bs = data['pose'].shape[0]
            
            # Single step prediction
            pusher_traj = torch.cat([data['pose'][:,:-1,None,-3:-1], data['pose'][:,1:,None,-3:-1]], dim=2).to(self.device).view(-1,2,2)
            state_0_gt = data['object_pose'][:,:-1,:, -3:-1].to(self.device).view(-1,n_particle,2)
            state_1_gt = data['object_pose'][:,1:,:, -3:-1].to(self.device).view(-1,n_particle,2)
            
            state_1_pred = self.model(state_0_gt, pusher_traj)
            for met in self.metric_ftns:
                value =  met(state_1_pred, state_1_gt).item()
                tracker.update(met.__name__+f'/single', value, n=bs)

            # Multi step prediction
            pusher_traj = data['pose'][:,:,-3:-1].to(self.device)
            state_0_gt = data['object_pose'][:,0,:, -3:-1].to(self.device)
            state_1_gt = data['object_pose'][:,-1,:, -3:-1].to(self.device)

            state_1_pred = self.model(state_0_gt, pusher_traj)
            for met in self.metric_ftns:
                value =  met(state_1_pred, state_1_gt).item()
                tracker.update(met.__name__+f'/multi', value, n=bs)

        self.empty_cache()
        return [tracker.avg(m.__name__+f'/{i}') for m in self.metric_ftns for i in ['single', 'multi']]
    
def balanced_loss(predict, gt_pred, gt_past, loss_fn, weight=0.05):

    with torch.no_grad():
        diff = (gt_pred - gt_past).norm(dim=-1)
        mask = torch.where(diff==0., torch.zeros_like(diff), torch.ones_like(diff)).unsqueeze(-1)

    loss_1 = loss_fn(mask*predict, mask*gt_pred)
    loss_0 = loss_fn((1.0-mask)*predict, (1-mask)*gt_pred)

    loss = loss_1 +loss_0*weight
    return loss

def train(model, cfg, name='mlp'):
    if cfg.train.seed:
        utils.set_seed(cfg.train.seed)
    
    train_filter, test_filter=None, None
    cache = False
    overfit = None#[2]#list(range(10))
    # overfit = [2]
    training_set = DynamicsDataset(mode='train',index_max = None, name_filter=train_filter, shuffle=False, seq_length = 17, data_dir=cfg.data_dir, cache = cache, cfg = cfg, overfit=overfit)
    testing_set = DynamicsDataset(mode='val', index_max = None, name_filter=test_filter, shuffle=False, seq_length=17, data_dir=cfg.data_dir, cache=cache, cfg = cfg, overfit=overfit)
    
    assert len(training_set) > 0 and len(testing_set) > 0, "make sure you have downloaded the dataset and set the correct path in the config file."
    # print(len(training_set), len(testing_set))

    batch_size = cfg.batch_size
    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, persistent_workers=True)
    val_dataloader = DataLoader(testing_set, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True)
    dataloader = {'train':train_dataloader, 'val':val_dataloader}

    # loss = torch.nn.L1Loss()
    loss = lambda pred,gt_pred, gt_past: balanced_loss(pred, gt_pred, gt_past, torch.nn.L1Loss(), weight=cfg.imbalance_ratio)
    # loss.__name__ = 'Error'

    met = torch.nn.L1Loss()
    met.__name__ = 'Error'

    # dataloader = DataLoader(training_set, batch_size=3, shuffle=True, num_workers=1, collate_fn=my_collate_all)
    # model = DynamicsPrediction(cfg)


    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.8, patience=3, verbose=True)

    if not name:
        name = datetime.now().strftime(r'%m%d_%H%M%S')
    
    
    trainer = Trainer(cfg, model, optim, dataloader, loss, met, scheduler=None, name=name, run_id="", comments="")

    return trainer.train()

def seq_pred(model, state, action):
    state = torch.tensor(state).float().unsqueeze(0).to(model.device)
    action = torch.tensor(action).float().unsqueeze(0).to(model.device)
    pred_list= []
    for t in range(1, action.shape[1]):
        pred_list.append(model(state[...,:], action[:,:t+1,:]).cpu().detach().squeeze(0).numpy())
    return np.array(pred_list)
        


def tsplot(data, x=None, **kwargs):
    if x is None:
        x = np.arange(data.shape[1])
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    plt.plot(x, mean, **kwargs)
    kwargs.pop('label')
    plt.fill_between(x, mean-std, mean+std, alpha=0.2, **kwargs)

    # plt.plot(x, np.quantile(data, 0.5, axis=0), **kwargs)
    # plt.fill_between(x, np.quantile(data, 0.9, axis=0), np.quantile(data, 0.1, axis=0), alpha=0.2, **kwargs)

def display_gif(gif_path):
    
    b64 = base64.b64encode(open(gif_path,'rb').read()).decode('ascii')
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))