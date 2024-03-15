import PIL
from IPython.display import Image
from matplotlib import cm
import numpy as np
import time
import torch
import os
# from pthflops import count_ops
from thop import profile
import random
import gc
import hydra
from omegaconf import OmegaConf

state_mean = 0.05
state_std = 0.5
action_mean = 0.01
action_std = 0.1

result_dir = './saved/results/'

def get_action(data,idx, use_dense_action=False):
    if use_dense_action:
        return data['dense_action'][:,idx,...]
    else:
        return data['action'][:,idx:idx+2,...]


def prediction(model, data, horizon=None, use_dense_action=False):
    if horizon is None:
        horizon=data['state'].shape[-3]
    if len(data['state'].shape)<=3:
        data = data.copy()
        for key in data:
            data[key] = data[key].unsqueeze(0)

    pred_result = [ data['state'][:,0,...].numpy() ]
    time_list = []
    model.eval()

    for i in range(horizon-1):
        action = get_action(data,i, use_dense_action).cuda()#.to(device)
        state_0_gt = data['state'][:,i:i+1,...].cuda()#.to(device)
        state_1_gt = data['state'][:,i+1:i+2,...].cuda()#.to(device)
        # print(action.device)
        # print(state_0_gt.device)
        # print(next(model.parameters()).device)
        
        t1 = time.time()
        if i==0:
            state_1_pred = torch.clip(model(state_0_gt,action), min=0.0, max=1.)
        else:
            state_1_pred = torch.clip(model(state_1_pred,action), min=0.0, max=1.)
        t2 = time.time()
        time_list.append(t2-t1)

        pred_result.append(state_1_pred.cpu().detach().numpy())
    # print(time_list)
    time_list = sorted(time_list)[2:-2]
    average_time = np.mean(time_list)
    print(f'Avg. running time: {average_time}s')

    # counts = count_ops(model, [state_0_gt, action])
    macs, params = profile(model, inputs=(state_0_gt, action))
    print(f'GFLOPS: {macs *2 / 1e9}')
    return pred_result

def test_time(model, dataset, horizon=None, use_dense_action=False, device = torch.device("cpu")):
    prev_device = next(model.parameters()).device
    # device = torch.device("cpu")
    model = model.to(device)
    if horizon is None:
        horizon=next(iter(dataset))['state'].shape[-3]

    time_list = []
    for data in dataset:
        if len(data['state'].shape)<=3:
            data = data.copy()
            for key in data:
                data[key] = data[key].unsqueeze(0)
        for i in range(horizon-1):
            action = get_action(data,i, use_dense_action).to(device)#.cpu()#.cuda()#.to(device)
            state_0_gt = data['state'][:,i:i+1,...].to(device)#.cpu()#.cuda()#.to(device)
            state_1_gt = data['state'][:,i+1:i+2,...].to(device)#.cpu()#.cuda()#.to(device)

            
            t1 = time.time()
            if i==0:
                state_1_pred = torch.clip(model(state_0_gt,action), min=0.0, max=1.)
            else:
                state_1_pred = torch.clip(model(state_1_pred,action), min=0.0, max=1.)
            t2 = time.time()
            time_list.append(1./(t2-t1))
    time_list = np.array(time_list)
    model = model.to(prev_device)

    print(f'Mean: {time_list.mean()}, Std: {time_list.std()}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compose_img(state,action):
    if type(state) is not type(np.array([1.])):
        state = state.cpu().detach().numpy()
    if type(action) is not type(np.array([1.])):
        action = action.cpu().detach().numpy()
    state = state.reshape((state.shape[-2],state.shape[-1],1))
    action = action.reshape((action.shape[-2],action.shape[-1],1))
#     state = state*state_std+state_mean
#     action = action*action_std+action_mean
# #     action = np.where(action<0.1,0,1)
#     action = np.clip(action,0.,1.)
#     print(action.max())
    img = np.concatenate([state, action,np.zeros_like(state)],axis=2)
    img = img.clip(0.,1.)
    return img
    pass
def get_error_img(state1,state2):
    if type(state1) is not type(np.array([1.])):
        state1 = state1.cpu().detach().numpy()
    if type(state2) is not type(np.array([1.])):
        state2 = state2.cpu().detach().numpy()
    # print(state2.shape)
    state1 = state1.reshape((state1.shape[-2],state1.shape[-1]))
    state2 = state2.reshape((state2.shape[-2],state2.shape[-1]))
    
    # img = cm.viridis((state1*state_std-state2*state_std+1)/2)
    img = cm.viridis(state1-state2)
    return img[...,:3]

def save_gif(img_list, path, display=True):
    for i,img in enumerate(img_list):
        img_list[i] = PIL.Image.fromarray(np.uint8(img*255))
#     img_list[0].save
    img_list[0].save(os.path.join(path+'.gif'), save_all=True, append_images=img_list[1:], duration=500, loop=0)
    if display:
        return Image(open(path+'.gif','rb').read(),width = 300)

def hstack_img(img_list,sep,gap_rgb=[255,255,255]):
    num_channel = img_list[0].shape[2]

    if img_list[0].max()<=1:
        gap_rgb=np.array(gap_rgb)/255.

    if num_channel==4:
        gap_rgb = np.hstack([gap_rgb,[1.]])
    
    line_new = []
    for i in range(len(img_list)):
        if i==0:
            w_gap_h = img_list[i].shape[0]
            w_gap = np.ones((w_gap_h,sep,num_channel))
            w_gap[...,:] = np.array(gap_rgb)
            w_gap = w_gap.astype(img_list[i].dtype)
            line_new.append(img_list[i])
            # print(img_list[i].shape)
        else:
            line_new.append(w_gap)
            line_new.append(img_list[i])
            # print(w_gap.shape)
            # print(img_list[i].shape)
    # print([s.shape for s in line_new])
    img_line = np.concatenate(line_new,axis=1)
    return img_line

def gen_eval_gif(model, data, name, display=True, use_dense_action=False):

    action = data['action']
    state = data['state']
    # img = compose_img(state_1_pred[0,0], action[0,0])
    pred_result = prediction(model, data, use_dense_action=use_dense_action)

    print(f'Number of params : {count_parameters(model)}')
    # print(max(pred_result))
    # plt.imshow(img)
    horizon=data['state'].shape[-3]
    img_list = []
    for i in range(horizon-1):
        img_error = get_error_img(pred_result[i],data['state'][i])
        img = compose_img(pred_result[i], action[i])
        img = hstack_img([img,img_error],1)
    #     img = compose_img(data['state'][idx,i], data['action'][idx,i])
        img_list.append(img)
    return save_gif(img_list, os.path.join(result_dir, name))

def get_gt_gif(data,name):
    action = data['action']
    state = data['state']
    horizon=data['state'].shape[-3]
    img_list = []
    for i in range(horizon-1):
        img = compose_img(state[i], action[i])
        img_list.append(img)
    return save_gif(img_list, os.path.join(result_dir, name))

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

def clear_cache():
    torch.cuda.empty_cache(); gc.collect()

def get_conf(path='ndf/cfg', name='eval'):
    c = []
    hydra.main(config_path=os.path.join('..','..', path), config_name=name, strict=False)(lambda x:c.append(x))()
    cfg = c[0]
    return cfg
def load_config(config_path):
    cfg= OmegaConf.load(config_path)
    cfg.root_dir='./'
    return cfg

# from hydra.experimental import compose, initialize
# with initialize(
def get_override_args(cfg):
    # overrides_path = hydra.utils.get_original_cwd() + '/.hydra/overrides.yaml'
    # # load the command line overrides file
    # overrides_cfg = OmegaConf.load(overrides_path)
    # overrides_cfg = OmegaConf.to_container(overrides_cfg, resolve=True)
    args_parser = hydra._internal.utils.get_args_parser()
    args = args_parser.parse_args().overrides
    keys = [aa.split('=')[0].split('.')[0] for aa in args]
    args = {key:cfg[key] for key in keys}
    return args

def test_get_overidden_conf(cfg):
    print(get_override_args(cfg))

def hashing(data, digit=6):
    if type(data) is torch.Tensor:
        data = data.detach().cpu().numpy()
    data = tuple(np.rint(data * 10**digit).reshape(-1).astype(int).tolist())
    return hash(data)

def get_object_poses(info):
    '''
    return N x 7
    '''
    def check_valid_block(k,v):
        if type(k) is not int:
            return False
        if k <= 0:
            return False
        if v[-1] == (0.006, 0.006, 5e-05):
            return False
        return True
    
    def get_pose(data_t):
        poses_t = []
        for k, p in sorted(data_t.items(), key = lambda x:x[0] if type(x[0]) is int else -1):
            if not check_valid_block(k ,p): continue
            # print(k)
            poses_t.append(list(p[1])+list(p[0]))
        return np.array(poses_t)
    
    if type(info) is list:
        results = [get_pose(data_t) for data_t in info]
    else:
        results = get_pose(info)

    return np.array(results).astype(np.float32)

if __name__ == '__main__':
    cfg = get_conf()
    test_get_overidden_conf(cfg)
    pass