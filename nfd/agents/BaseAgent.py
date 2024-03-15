import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
from nfd.utils.renderer import render_field
from cliport import tasks
# import nfd.models.model as model
# from cliport.models.prediction import TransitionFCN

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle as pkl
from nfd.utils.image_utils import GIFSaver
import gc
import re
import inspect

# from nfd.models.model import 
import nfd.models
sys.path.append(os.path.dirname(nfd.models.__file__))
# from nfd.models.prediction import Prediction
# model.TransitionFCN = model.FCN
from datetime import datetime
# from nfd.utils.log_utils import save_codes, save_cfg
from nfd.utils.utils import get_object_poses

# PUSHER_FIELD = 'ur5/spatula/spatula-rasterization.npy'
# ZONE_FIELD = 'zone/zone-rasterization.npy'
# ZONE_FIELD_SDF = 'zone/zone-rasterization-sdf.npy'

def get_state(color, pooling=False):
    states = color[...,0].float()
    vmax = 200.
    vmin = 70.
    states = torch.clamp(states, min = vmin, max=vmax)
    states = (states-vmin)/(vmax-vmin)
    if pooling:
        states = pooling(states)
    return states

def get_action(color, pooling=False):
    vmax = 200.
    vmin = 100.
    actions = color[...,2].float()
    actions = torch.clamp(actions, min = vmin, max=vmax)
    actions = (actions-vmin)/(vmax-vmin)
    # actions = (actions>128).to(actions.dtype)
    if pooling:
        actions = pooling(actions)
    return actions

def get_goal(color, pooling=False):
    vmax = 200.
    vmin = 100.
    goals = color[...,1].float()
    goals = torch.clamp(goals, min = vmin, max=vmax)
    goals = (goals-vmin)/(vmax-vmin)
    # goals = (goals>128).to(goals.dtype)
    if pooling:
        goals = pooling(goals)
    return goals

def sdf_loss(states, goal_sdf):
    original_shape = states.shape
    # batch_size = states.shape[0]
    # mask = torch.clamp(goal_sdf,0,0.1)*10
    loss = (states * goal_sdf).view(-1, original_shape[-1]*original_shape[-2]).sum(axis=1)# + (states * mask).view(batch_size, -1).sum(axis=1)

    return loss.view(original_shape[:-2])

def success_rate(states, goal_sdf):
    batch_size = states.shape[0]
    threshold1 = 0.015
    threshold2 = 0.02
    mask = (torch.clamp(goal_sdf,threshold1,threshold2)-threshold1)*1/(threshold2-threshold1)
    # plt.figure()
    # plt.imshow(mask[0,...])
    # # plt.imshow(states[0,...])
    # plt.show()
    loss = (states * mask).view(batch_size, -1).sum(axis=1)/states.view(batch_size, -1).sum(axis=1)
    
    return 1-loss

Z0 = 0.005

class Base():
    def __init__(self, cfg):
        self.cfg = cfg

        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array(cfg.bounds)
        self.pix_size = cfg.pix_size
        H = int(round((self.bounds[1][1]-self.bounds[1][0])/self.pix_size))
        W = int(round((self.bounds[0][1]-self.bounds[0][0])/self.pix_size))
        self.in_shape = (H, W, 6)
        self.margin = cfg.agent_margin

        self.assets_root = cfg['assets_root']
        self.dtype = torch.float
        self.auto_empty_cache = False

        self.load_assets()
        if cfg['down_sample']!=1:
            self.pre_process = nn.AvgPool2d(cfg['down_sample'], stride=cfg['down_sample'])
        else:
            self.pre_process = nn.Identity
        self.to(torch.device('cuda' if cfg['gpu'] else 'cpu'))

        self.get_state = get_state
        self.loss_eval = sdf_loss
        self.reward_eval = success_rate

        self.result_data = None


    def to(self, device):
        self.device=device
    
    def set_seed(self, seed=None):
        if seed:
            utils.set_seed(seed)
    def rendering(self, pose, field):
        '''
        pose: NxTx7 torch array
        '''
        # field_bk = field
        height, width = self.in_shape[0], self.in_shape[1]
        original_pose_shape = pose.shape
        # field, bounds_field= field['data'].to(pose), field['bounds']
        if len(pose.shape)==1:
            pose = pose.unsqueeze(0)

        if len(pose.shape)==3:
            pose = pose.view(-1, 7)
        output = render_field(pose, field, self.heightmap_points)
        self.empty_cache()
        if len(original_pose_shape)==3:
            # ##
            # output_ref = render_field(pose, field_bk, height, width, self.heightmap_points)
            # print(pose.shape)
            # ##
            # print(torch.equal(output, output_ref))
            output = output.view(original_pose_shape[0], original_pose_shape[1], height, width)
            
            return output
        else:
            return output[:,0,...]
    def load_assets(self):
        # zone_path = self.path
        # self.pusher_field = np.load(os.path.join(self.assets_root, PUSHER_FIELD),allow_pickle=True).item()
        self.pusher_field = np.load(os.path.join(self.cfg['pusher_field_dir'], self.cfg['pusher_field'] ),allow_pickle=True).item()
        self.pusher_field['data'] = torch.tensor(self.pusher_field['data'].astype(np.float32))
        # self.zone_field = np.load(os.path.join(self.assets_root, ZONE_FIELD_SDF),allow_pickle=True).item()
        self.zone_field = np.load(self.cfg['zone_field'],allow_pickle=True).item()
        self.zone_field['data'] = torch.tensor(self.zone_field['data'].astype(np.float32))

        height, width = self.in_shape[0], self.in_shape[1]
        pix_size = (self.bounds[0,1] - self.bounds[0,0])/width
        
        
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        # print(px.shape)
        # print(py.shape)
        px = pix_size*px+self.bounds[0,0]
        py = pix_size*py+self.bounds[1,0]
        
        # points: HxWx2 xy coordinates of each pixel in heightmap
        self.heightmap_points = np.concatenate([px[...,None], py[...,None]], axis=2)

    def empty_cache(self, force=False):
        if force or self.auto_empty_cache:
            torch.cuda.empty_cache(); gc.collect()

    def clamp(self, points, margin = None):
        if margin is None:
            margin = self.margin
        '''
        points: ... x 2
        '''
        with torch.no_grad():
            xlim = self.bounds[0].copy()
            ylim = self.bounds[1].copy()
            xlim[0] = xlim[0]+margin
            xlim[1] = xlim[1]-margin
            ylim[0] = ylim[0]+margin
            ylim[1] = ylim[1]-margin

            points[...,0] = torch.clamp(points[...,0], xlim[0], xlim[1])
            points[...,1] = torch.clamp(points[...,1], ylim[0], ylim[1])

        return points
    
    def within_bounds(self, points, margin=0.025):
        '''
        points: ... x 2
        '''
        # xlim = self.bounds[0].copy()
        # ylim = self.bounds[1].copy()
        # xlim[0] = xlim[0]+margin
        # xlim[1] = xlim[1]-margin
        # ylim[0] = ylim[0]+margin
        # ylim[1] = ylim[1]-margin

        xlim = self.bounds[0] + np.array([margin, -margin])
        ylim = self.bounds[1] + np.array([margin, -margin])

        mask = torch.ones(*points.shape[:-1], dtype=bool)
        mask *= torch.where(points[...,0]>=xlim[0], 1., 0.).bool()
        mask *= torch.where(points[...,0]<=xlim[1], 1., 0.).bool()
        mask *= torch.where(points[...,1]>=ylim[0], 1., 0.).bool()
        mask *= torch.where(points[...,1]<=ylim[1], 1., 0.).bool()
        return mask

    def obs_process(self, obs, info, goal=None):
        zone_pose, _ = info['goal_zone']
        zone_pose = torch.tensor(np.hstack([zone_pose[1],zone_pose[0]])).float().to(self.device)

        with torch.no_grad():
            goal_zone = self.pre_process(self.rendering(zone_pose, self.zone_field))

        

        img = self.get_image(obs)
        
        color = torch.tensor(img[...,:3])
        self.obs = color/255.
        # depth = torch.tensor(img[...,3])
        state = self.get_state(color).to(self.device)
        return state, color, goal_zone
    def info_process(self, info):
        object_poses = torch.tensor(get_object_poses(info)).to(self.device,self.dtype)
        goal = info['goal_zone']

        pos = goal[0][0]
        rot = goal[0][1]
        goal_pose = torch.tensor(list(pos))[-3:-1].to(self.device,self.dtype)
        goal_size = torch.tensor(goal[1][0:2]).to(self.device,self.dtype)
        return object_poses, goal_pose, goal_size[0]
    
    def get_image(self, obs, cam_config=None):
        if cam_config is None:
            cam_config = self.cam_config

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img
    
    def get_goal_zone(self, info):
        zone_pose, _ = info['goal_zone']
        zone_pose = torch.tensor(np.hstack( [zone_pose[1],zone_pose[0]])).to(self.dtype)#.to(self.device)

        goal_zone = self.pre_process(self.rendering(zone_pose, self.zone_field))
        
        return goal_zone
    
    def get_loss(self, obs, info, goal=None):
        goal_zone = self.get_goal_zone(info)

        if type(obs) is dict:
            img = self.get_image(obs)
            color = torch.tensor(img[...,:3])
            state = self.get_state(color)
        else:
            state = obs
        state = self.pre_process(state.unsqueeze(0)).to(goal_zone)
        # print(goal_zone.shape)
        # print(state.shape)
        # input()
        with torch.no_grad():
            cost = self.loss_eval(state, goal_zone)
        return cost.cpu().numpy().item()  
    def get_reward(self, obs, info, goal=None):
        object_pose, goal_pos, goal_size = self.info_process(info)
        r_obj = 0.025
        r_target = goal_size/2

        d1 = r_target-r_obj
        d0 = r_target+r_obj

        r_min = 0.01

        
        dist = (object_pose[:,[-3,-2]] - goal_pos.view(1,-1)).norm(dim=-1)
        rr = torch.clamp(dist, d1, d0)
        reward = 1- (rr-d1)/(d0-d1)
        return reward.mean().item()

        # goal_zone = self.get_goal_zone(info)

        # img = self.get_image(obs)
        # color = torch.tensor(img[...,:3])
        # state = self.pre_process(self.get_state(color).unsqueeze(0))

        # with torch.no_grad():
        #     reward = self.reward_eval(state, goal_zone)
        # return reward.numpy().item()
    
    def pos2uv(self, pos):
        x, y = pos
        u = (x - self.bounds[0, 0]) / self.pix_size
        v = (y - self.bounds[1, 0]) / self.pix_size
        return u, v

    def uv2pos(self, uv):
        u, v = uv
        x = u * self.pix_size + self.bounds[0, 0]
        y = v * self.pix_size + self.bounds[1, 0]
        return x, y
    
    def save_result(self, obs, color, state, info, goal, iter_history=None):

        result_dir = os.path.join(self.cfg.root_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(os.path.join(result_dir), exist_ok=True)
        
        if self.result_data is None:
            # save_cfg(os.path.join(result_dir, 'checkpoint.yaml'), self.cfg)
            now = datetime.now()
            self.result_id = now.strftime(f"%m-%d-%H-%M-%S")
            print('Saving to: ', os.path.join(result_dir, self.result_id+'.pkl'))
 
            self.result_data = {'cfg':self.cfg, 'raw_img':[], 
                                'color':[], 'state':[], 'goal':goal, 'iter_history':[],
                                'trajectory':[], 'loss':[], 'info': info}

        # result_file = os.path.join(result_dir, str(self.result_current_time).zfill(6)+'.pkl')
        result_file = os.path.join(result_dir, self.result_id+'.pkl')
        
        # list all files in the directory, and match with an reg expression

        # exist_num = [int(f.split('.')[0]) for f in os.listdir(result_dir) if os.path.isfile(os.path.join(result_dir, f)) and re.match(r"\d{6}\.pkl$", f)]
        # if len(exist_num)>0:
        #     num = max(exist_num)+1
        # else:
        #     num = 1
        # filename = os.path.join(result_dir, str(num).zfill(6)+'.pkl')
        self.result_data['raw_img'].append(obs)
        self.result_data['color'].append(color.detach().cpu().numpy())
        self.result_data['state'].append(state.detach().cpu().numpy())
        self.result_data['trajectory'].append(self.trajectory)
        self.result_data['loss'].append(self.get_loss(state, info))
        self.result_data['iter_history'].append(iter_history)
        # data = {'cfg':self.cfg, 'obs':obs, 'color':color, 'state':state, 'goal':goal, 'traj':self.trajectory}
        # self.result_history.append(data)
        with open(result_file, 'wb') as f:
            pkl.dump(self.result_data, f)
        # self.result_current_time += 1

    def disp_result(self, color, state, info, trajs, costs=None, axs=None):
        if axs is None:
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
        else:
            ax1, ax2 = axs
        
        ax1.clear()
        self.visualize_obs(color, ax=ax1)
        self.visualize_trajs(trajs, costs=costs, ax=ax1)
        
        zone_pose, _ = info['goal_zone']
        zone_pose = torch.tensor(np.hstack([zone_pose[1],zone_pose[0]])).float().to(self.device)
        goal_sdf = self.rendering(zone_pose, self.zone_field)
        goal_mask = torch.zeros_like(goal_sdf)
        goal_mask[torch.where(goal_sdf==0)]=1.
        self.visualize_obs(torch.cat([state[...,None], goal_mask[0,...,None], torch.zeros_like(state[...,None])], axis=2), ax=ax2)
        
    
    def visualize_trajs(self,trajs,show_rot=False, costs=None,title=None,ax=None):
        '''
        trajs: NxTx7
        '''
        if ax is None:ax=plt.gca()
        
        if torch.is_tensor(trajs):
            trajs = trajs.detach().cpu().numpy()
        if trajs.shape[-1]==2:
            zzz = np.zeros_like(trajs[...,[0]])
            trajs = np.concatenate([zzz, zzz, zzz, zzz+1,trajs, zzz], axis=-1)

        if not (costs is None):
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes('right', size='5%', pad=0.05)

            costs = np.array(costs)
            # costs = np.clip(costs, 0, 2.)
            costs = 1-(costs-costs.min()+1e-5)/(costs.max()-costs.min()+1e-5)
            colors = plt.cm.plasma(costs)
            # print(colors)
        else:
            colors = [[0.7,0.7,1.0]]*trajs.shape[0]

        for i in range(trajs.shape[0]):
            traj = trajs[i,...]
            ax.plot(traj[:,4],traj[:,5],"-", marker='.', linewidth=0.5, markersize=2,mfc='none',color=colors[i])
            ax.arrow(traj[-2,4], traj[-2,5], (traj[-1,4]-traj[-2,4])/2, (traj[-1,5]-traj[-2,5])/2, color=colors[i], head_width=0.02, head_length=0.02, overhang=1.0)
            # ax.arrow(traj[-3,4], traj[-3,5], (traj[-2,4]-traj[-3,4])/2, (traj[-2,5]-traj[-3,5])/2, color=colors[i], head_width=0.03, overhang=1.0)
            if show_rot:
                arrow_length = 0.05
                for j in range(traj.shape[0]):
                    pose = traj[j,:]
                    theta = utils.quatXYZW_to_eulerXYZ((pose[:4].tolist()))[-1]
                    ax.plot([pose[4],pose[4]+arrow_length*np.cos(theta)], [pose[5],pose[5]+arrow_length*np.sin(theta)], 'k')
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        if title:
            ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
    
    def visualize_points(self, points, ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        # extent = self.bounds[:2,:].reshape(-1)
        ax.scatter(points[:,0], points[:,1], **kwargs)
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.set_aspect('equal', adjustable='box')

    
    def visualize_obs(self,obs=None,ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        extent = self.bounds[:2,:].reshape(-1)
        if obs is None:
            ax.imshow(self.obs, extent=extent,origin='lower')
        else:
            if type(obs) is torch.Tensor:
                obs = obs.detach().cpu().numpy()
            if not np.issubdtype(obs.dtype, np.integer) and obs.max()>1.:
                obs = obs/255.
            ax.imshow(obs, extent=extent,origin='lower', **kwargs)
        ax.set_aspect('equal', adjustable='box')
    
    def visualize_field(self, pose,field, ax=None):
        if ax is None: ax = plt.gca()
        pose = torch.tensor(pose).view(1,7)
        img = self.rendering(pose, field)
        img = img.detach().cpu().numpy()[0,...]
        extent = self.bounds[:2,:].reshape(-1)
        ax.imshow(img, extent=extent,origin='lower')

    def visualize_all(self, state, info, obstacle=None, ax=None):
        if ax is None: ax = plt.gca()

        zone_pose, _ = info['goal_zone']
        zone_pose = torch.tensor(np.hstack([zone_pose[1],zone_pose[0]])).float().to(self.device)
        with torch.no_grad():
            goal_sdf = self.rendering(zone_pose, self.zone_field)

            goal_mask = torch.zeros_like(goal_sdf)
            goal_mask[torch.where(goal_sdf==0)]=1.
            goal_mask = goal_mask.detach().cpu()[0,...]
        
        state = torch.tensor(state)
        if obstacle is None:
            obstacle = torch.zeros_like(state)
        
        rgb = torch.cat([state[...,None], goal_mask[...,None], obstacle[...,None]], axis=2)

        extent = self.bounds[:2,:].reshape(-1)
        ax.imshow(rgb.detach().cpu().numpy(), extent=extent, origin='lower')
        # ax.set_aspect('equal', adjustable='box')
        
class BaseAgent(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        # utils.set_seed(0)

        # self.pred = Prediction(cfg)
        self.pred = None

        self.to(torch.device('cuda' if cfg['gpu'] else 'cpu'))
        # self.pix_size = 0.003125
        # self.in_shape = (320, 160, 6)
        # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        if 'n_sample' in cfg: self.n_sample = cfg.n_sample
        self.horizon = cfg.horizon
        
        # if "model_path" in cfg and cfg.model_path and cfg.set_model:
        #     self.set_model(cfg['model_path'])

        self.trajectory = []
        self.im_traj = [] # Intermediate traj
        
        if 'verbose' in cfg:
            self.verbose=cfg.verbose
    
        
    
    def set_model(self, model_path):
        if model_path:
            self.pred.set_model(model_path)


import matplotlib as mpl

def plot_layer_image(img1,img2,ax=None,cb=False,**arg):
    if ax is None:ax = plt.gca()
    iii = plt.imshow(img1.detach().cpu(),**arg)
    if cb:
        plt.colorbar()
    cmap = mpl.cm.get_cmap('Reds')
    img2 = img2.detach().cpu()
    img2 = img2/img2.max()
#     img2 = cmap(img2.detach().cpu())
    zzz = torch.zeros_like(img2[...,None])
    img2 = torch.cat([img2[...,None],zzz,zzz,img2[...,None]],axis=2)
    plt.imshow(img2,**arg)
    return iii