"""Ravens main training script."""

import os
import pickle as pkl
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from multiprocessing import Process

import time

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment, PilesEnvironment
from cliport.tasks import cameras
from cliport.dataset import RavensDataset
# from nfd.agents.LineAgent import LineShootAgent, LineOptAgent, gen_pose_list_batch
# # from nfd.agents.DPIAgent import DPIShootAgent, DPIOptAgent
# from nfd.agents.random_agent import RandomAgent, RandomShootingAgent
# from nfd.agents.ObstacleAvoidanceAgent import ObstacleAvoidanceAgent, get_obstacle
# from nfd.agents.ndf_agent import NDFAgent
from nfd.agents.BaseAgent import BaseAgent
# from nfd.agents.curve_agent import CurveAgent as CurveAgent_bk
# from nfd.agents.CurveAgent import CurveAgent
# from nfd.agents.SplitAgent import SplitAgent, SpreadAgent
# from nfd.agents.ObjectCentricAgent import ObjectCentricAgent
# from nfd.toy.agent_toy import ToyShootAgent, ToyOptAgent
# from nfd.utils.renderer import render_field
# from nfd.gen_data import gen_target_list
from nfd.utils.planning_utils import gen_action_list
import matplotlib.pyplot as plt
import pickle as pkl

from nfd.utils.image_utils import GIFSaver
from tqdm import tqdm

cam_config = cameras.RealSenseD415.CONFIG
bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
pix_size = 0.003125


# model_dir = './saved/models/'

class Evaluator(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # print('disp', cfg.disp)
        # input()
        env = PilesEnvironment(
            cfg['assets_root'],
            disp=cfg['disp'],
            shared_memory=cfg['shared_memory'],
            hz=480,
            # enableGPU = cfg['sim_enableGPU'],
            record_cfg=cfg['record'],
            **cfg.env
            # object_case=cfg.object_case,
            # pusher_case=cfg.pusher_case,
        )
        self.env = env
        self.cfg = cfg
        task = tasks.names[cfg['task']]()
        self.task = task
        task.mode = cfg['mode']
        # record = cfg['record']['save_video']
        save_data = cfg['save_data']
        self.max_retry = cfg.max_retry
        self.abstract_action = False

        self.auto_empty_cache = True

        self.agent_oracle = task.oracle(env)
        self.agent_eval = self.agent_oracle#RandomShootingAgent(cfg)
        # data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
        # dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
        self.threshold = 0.2

    def set_agent(self, agent):
        self.agent_eval = agent
        
    def eval(self, agent=None, model_path=None, N=10, name=None):
        
        if agent:
            agent_eval = agent
            if self.cfg.agent=="transporter":
                self.abstract_action = True
        else:
            agent_eval = self.agent_oracle
            self.abstract_action = True
        if model_path:
            agent_eval.set_model(model_path)
        
        if name is None:
            name = agent_eval.__class__.__name__

        # self.save_dir = os.path.join(self.cfg['save_dir'], name)
        self.save_dir = self.cfg['save_dir']
        if self.cfg.save_data and ( not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        
        # input(agent.name)
        # if type(agent) is SplitAgent or type(agent) is SpreadAgent or agent.name.lower().startswith('split') or agent.name.lower().startswith('spread'):
        #     # input('split')
        #     self.get_goal_zone = agent.get_goal_zone
        #     self.loss_eval = agent.loss_eval
        #     self.reward_eval = agent.reward_eval

        
        env = self.env
        task = self.task
        
        if type(N) == type(2):
            n_list = list(range(N))
        else:
            n_list = N
        # seed = -1 + 10000 
        metrics_list = []
        for n in tqdm(n_list):
            # while True:
            #     try:
            #         self.eval_one(n, env, task, agent_eval)
            #     except RuntimeError as e:
            #         print(repr(e))
            #         # raise e
            #         time.sleep(30)
            #         continue
            #     else:
            #         break
            metrics =  self.eval_one(n, env, task, agent_eval)
            metrics_list.append(metrics)
        
        metrics_overall = {met: np.array([metrics[met] for metrics in metrics_list]) for met in metrics_list[0].keys()}
        return metrics_overall
            # seed += 2
    def check_collision(self, pose0, pose1, obs):
        pass
    #     pos0 = np.float32((pose0[0][0], pose0[0][1]))
    #     pos1 = np.float32((pose1[0][0], pose1[0][1]))
    #     vec = np.float32(pos1) - np.float32(pos0)
    #     length = np.linalg.norm(vec)
    #     vec = vec / length
    #     pos0 -= vec * 0.02
    #     pos1 -= vec * 0.05
    #     # theta = np.arctan2(vec[1], vec[0])
    #     # rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    #     img = self.get_image(obs)
    #     color = torch.tensor(img[...,:3])
    #     obstacle = get_obstacle(color).to(self.device)
    #     endpoints = torch.tensor([pos0, pos1])[None,...].to(self.device)
    #     # print(endpoints)
    #     trajs = gen_pose_list_batch(endpoints, horizon=self.horizon)
    #     # print(trajs)
    #     actions = self.rendering(trajs, self.pusher_field)[0,...]
    #     action_cost = (actions* obstacle[None,...]).sum(axis=[-1,-2])
    #     # print(action_cost.shape)
    #     invalid_idx = torch.where(action_cost>self.threshold)[0]
    #     if invalid_idx.shape[0]:
    #         max_idx = invalid_idx[0] - 1
    #         # valid_trajs = trajs[:max_idx,...]
    #         pad = self.horizon - max_idx.item() +2
    #         valid_trajs = trajs[0,list(range(max_idx))+[max_idx-1]*pad, ...]
    #         # print(pad)
    #     else:
    #         valid_trajs = trajs[0,...]
    #     # print(valid_trajs.shape[0])
    #     return gen_action_list(valid_trajs)
        
    
    def eval_one(self, n, env, task, agent_eval):
        seed = n*2+ 10001
        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        gif = GIFSaver()
        env.set_task(task)
        env.seed(seed)
        obs = env.reset()
        img = self.get_image(obs)
        info = env.info
        # reward = 0
        # loss_init = agent_eval.get_loss(obs, info)
        # print('caonima')
        
        time_spent={'plan':0., 'sim':0}
        # print(self.cfg['max_steps'])
        metrics = {
            "cost":[],
            "reward_gt": [],
            "reward": []
        }

        # loss_hist = []
        metrics["cost"].append(self.get_loss(obs, info))
        metrics["reward"].append(self.get_reward(obs, info))
        metrics["reward_gt"].append(env.task.get_reward())
        color_hist = []
        traj_hist = []
        done=False

        for k in range(self.cfg['max_steps']):
            # print("Case:",n,"Push:", k, 'starts planning,', 'reward {metrics["reward"][-1]:.3f}')
            # print("Case:",n,"Push:", k, 'starts planning,', f'current loss {metrics["loss"][-1]/metrics["loss"][0]:.3f}, reward {metrics["reward"][-1]:.3f}')
            t0 = time.time()
            while True:
                fails = 0
                try:
                    if not done:
                        object_pose, goal_pos, goal_size = agent_eval.info_process(info)
                        traj = agent_eval.trajectory_generation(object_pose[...,[-3,-2]], goal_pos, goal_size)
                        target_list = gen_action_list(traj)
                        # traj_hist.append(agent_eval.trajectory)
                    else:
                        target_list = []#[False]*(self.cfg.horizon +4)
                except RuntimeError as e:
                    fails+=1
                    self.empty_cache()
                    if fails>=self.max_retry:
                        raise e
                    print(repr(e))
                    time.sleep(10*torch.rand(1).item())
                    continue
                else:
                    break
            t1 = time.time()
            # print("Case:",n,"Push:", k, 'ends planning')
            time_spent['plan']+=t1-t0

            self.empty_cache()

            
            for i,target in enumerate(target_list):
                t0 = time.time()
                if target:
                    obs, reward, _, info = env.step(target)
                    done = True if self.get_reward(obs, info) >= 0.999 else False
                t1 = time.time()
                time_spent['sim']+=t1-t0
                # img = self.get_image(obs)
                # color = torch.tensor(img[...,:3])
                # color_hist.append(color.numpy().astype(np.uint8))
                # plt.imshow(color.numpy().astype(np.int))
                # plt.pause(0.001)
                # gif.add(color)
                gif.add(obs['color'][0])
            
            metrics["cost"].append(self.get_loss(obs, info))
            metrics["reward"].append(self.get_reward(obs, info))
            metrics["reward_gt"].append(env.task.get_reward())
            # print(metrics["reward_gt"])
            # print("Case:",n,"Push:", k, f'reward {metrics["reward"][-1]:.3f}')

        
        # color_hist = np.array(color_hist, dtype=np.uint8)

        # data = {
        #     "color": color_hist,
        #     "trajectory": traj_hist,
        #     "info": info
        # }
        # data.update(metrics)

        self.save_dir = os.path.join(self.cfg.save_dir, agent_eval.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        if self.cfg.save_data:
            # with open(os.path.join(self.save_dir, f'{n:06d}-{seed}.pkl'),"wb") as f:
            #     pkl.dump(data,f)
            gif.save(name = f'{n:06d}-{seed}.gif', path = self.save_dir, duration=150)
        
        # loss = agent_eval.get_loss(obs, info)
        time_spent['plan']=time_spent['plan']/self.cfg['max_steps']
        time_spent['sim']=time_spent['sim']/self.cfg['max_steps']
        # print(n, time_spent)
        # 
        return metrics
    

# @hydra.main(config_path='../cfg', config_name='eval', strict=False)
# def main(cfg):
#     ev = Evaluator(cfg)
#     # 
#     # name = 'fcnrender_bn_lnd_ms'
#     # name = 'fcnrender_bn_rnd_ms'
#     name = 'fcnrender_bn_1k_ms'
#     # name = cfg["model_name"]
#     # eval.eval(model_path = f'./models/{name}_model.pt', name=name+'_shooting_n128')
#     # eval.eval(model_path = f'./saved/models/{name}/model_best.pt', name=name+'_shooting_n128')
#     # eval.eval(agent = NDFAgent(name+'_adam_lr1e-2_n32_ep30', cfg))

#     # eval.eval_multi(model_path = f'./saved/models/{name}/model_best.pt', name=name+'_shooting_n128')
#     if cfg.agent:
#         agent = eval(cfg.agent)
#         agent = agent(cfg)
#     elif cfg["agent_type"].lower().startswith("shoot"):
#         agent = RandomShootingAgent(cfg)
#     elif cfg.agent_type.lower().startswith("opt"):
#         agent = NDFAgent( cfg,"ndf", seed=0)
#     elif cfg.agent_type.lower().startswith("curve"):
#         agent = CurveAgent_bk(cfg)
#     elif cfg.agent_type.lower() == "oracle":
#         agent = None
#     else:
#         raise Exception("Unknown agent type %s"%cfg["agent_type"])
#         # print(cfg["agent_type"])
#         # input()
#         return
#     model_path = cfg["model_path"]
#     # print(model_path)
#     # input()
#     if agent:
#         agent.set_model(model_path)
#     cfg["save_dir"] = os.path.join(cfg.save_dir, )
#     ev.eval(agent=agent, N=cfg["test_idx"])

# if __name__ == '__main__':
#     main()
#     # test_curve_agent()