import numpy as np
import torch
from cliport.utils import utils

Z0 = 0.005

def gen_action_list3(pose_list):
    '''
    convert trajectory to env action
    '''
    print(pose_list[:,2])
    print(pose_list[:,3])
    input()
    # print(pose0)
    # print(pose1)
    pos0 = pose_list[0][4:]
    pos1 = pose_list[-1][4:]
    rot0 = pose_list[0][:4]
    rot1 = pose_list[-1][:4]
    target_list = []
    # print(pos0)
    # print(pos1)
    zero_rot = [0., 0., 0., 1.]

    over0 = (pos0[0], pos0[1], 0.31)
    over1 = (pos1[0], pos1[1], 0.31)
    over1_0 = (pos1[0], pos1[1], 0.05)

    # Execute push.
    target_list.append({"pos": over0, "rot":rot0})
    target_list.append({"pos": pos0, "rot":rot0})

    #n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
    if  len(pose_list) ==2:
        horizon=16
        for i in range(1, horizon+1):
            target = pos0 * (horizon-i)/float(horizon) + pos1 * i/float(horizon)
            target_list.append({"pos": target, "rot":rot1, "speed":0.001})
    else:
        for pose in pose_list[1:]:
            target = pose[4:]
            rot = pose[:4]
            # target = pos0 + vec * i * 0.01
            # print(target)
            target_list.append({"pos": target, "rot":rot, "speed":0.001})
            # timeout |= movep((target, rot), speed=0.003)
    # target_list.append({"pos": pos1, "rot":rot1, "speed":0.001})
    target_list.append({"pos": over1_0, "rot":rot1, "speed":0.001})
    target_list.append({"pos": over1, "rot":zero_rot})
    # input()
    return target_list

def gen_action_list(pose_list):
    '''
    convert trajectory to env action
    '''
    # print(pose_list)
    # print(pose0)
    # print(pose1)
    # pose_list = pose_list[:,[-3,-2]]
    
    # pose_list = pose_list.detach()
    pose_list = np.concatenate([pose_list, np.zeros_like(pose_list[:,[0]])+Z0], axis=1)

    x0 = pose_list[0, 0]
    y0 = pose_list[0, 1]
    x1 = pose_list[-1, 0]
    y1 = pose_list[-1, 1]
    # z = np.zeros_like(x0)+Z0
    # q1 = np.zeros_like(x0)
    # q2 = np.zeros_like(x0)
    theta = np.arctan2(y1-y0,x1-x0)
    
    q0 = np.cos(theta/2)
    q3 = np.sin(theta/2)

    # rot0 = np.vstack([q1, q2, q3, q0]).view(4).detach().cpu().numpy()
    rot0 = np.array([0., 0., q3, q0], dtype=pose_list.dtype)
    # import pdb; pdb.set_trace()
    # pose_list = pose_list.detach().cpu().numpy()
    
    pos0 = pose_list[0]#[4:]
    pos1 = pose_list[-1]#[4:]
    # rot0 = rot#pose_list[0][:4]
    rot1 = rot0#pose_list[-1][:4]
    target_list = []
    # print(pos0)
    # print(pos1)
    zero_rot = [0., 0., 0., 1.]

    over0 = (pos0[0], pos0[1], 0.31)
    over1 = (pos1[0], pos1[1], 0.31)
    over1_0 = (pos1[0], pos1[1], 0.05)

    # Execute push.
    target_list.append({"pos": over0, "rot":rot0})
    target_list.append({"pos": pos0, "rot":rot0})

    #n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
    if  len(pose_list) ==2:
        horizon=16
        for i in range(1, horizon+1):
            target = pos0 * (horizon-i)/float(horizon) + pos1 * i/float(horizon)
            target_list.append({"pos": target, "rot":rot1, "speed":0.001})
    else:
        for pose in pose_list[1:]:
            target = pose#[4:]
            rot = rot0#pose[:4]
            # target = pos0 + vec * i * 0.01
            # print(target)
            target_list.append({"pos": target, "rot":rot, "speed":0.001})
            # timeout |= movep((target, rot), speed=0.003)
    # target_list.append({"pos": pos1, "rot":rot1, "speed":0.001})
    target_list.append({"pos": over1_0, "rot":rot1, "speed":0.001})
    target_list.append({"pos": over1, "rot":zero_rot})
    # input() 
    # print(target_list)
    # input()

    return target_list

def gen_action_list3(pose_list):
    '''
    convert trajectory to env action
    '''
    # print(pose_list)
    # print(pose0)
    # print(pose1)
    pose_list = pose_list[:,[-3,-2]]
    pose_list = pose_list.detach()


    
    pose_list = torch.cat([pose_list, torch.zeros_like(pose_list[:,[0]])+Z0], dim=1)

    pos0 = pose_list[0,:]
    pos1 = pose_list[1,:]
    # print('pos1',pos0)
    vec = pos1 - pos0

    theta = torch.atan2(vec[1], vec[0])
    # rot0 = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    q0 = torch.cos(theta/2)
    q3 = torch.sin(theta/2)
    rot1 = rot0 = torch.vstack([0., 0., q3.item(), q0.item()], dtype=np.float32)
    
    pose_list = pose_list.numpy()
    pos0 = pos0.numpy()
    pos1 = pos1.numpy()
    # rot1 = rot0
    # rot0 = pose_list[0][:4]
    # rot1 = pose_list[-1][:4]
    target_list = []
    # print(pos0)
    # print(pos1)
    zero_rot = [0., 0., 0., 1.]
    # rot1 = rot0 = zero_rot
    

    over0 = (pos0[0], pos0[1], 0.31)
    over1 = (pos1[0], pos1[1], 0.31)
    over1_0 = (pos1[0], pos1[1], 0.05)

    # Execute push.
    target_list.append({"pos": over0, "rot":rot0})
    target_list.append({"pos": pos0, "rot":rot0})

    #n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
    if  len(pose_list) ==2:
        horizon=16
        for i in range(1, horizon+1):
            target = pos0 * (horizon-i)/float(horizon) + pos1 * i/float(horizon)
            target_list.append({"pos": target, "rot":rot1, "speed":0.001})
    else:
        for pose in pose_list[1:]:
            target = pose#[4:]
            rot = rot0#pose[:4]
            # target = pos0 + vec * i * 0.01
            # print(target)
            target_list.append({"pos": target, "rot":rot, "speed":0.001})
            # timeout |= movep((target, rot), speed=0.003)
    # target_list.append({"pos": pos1, "rot":rot1, "speed":0.001})
    target_list.append({"pos": over1_0, "rot":rot1, "speed":0.001})
    target_list.append({"pos": over1, "rot":zero_rot})
    # input()
    # print(target_list)
    # input()
    return target_list

def planning_loss(states, goal_sdf, soft_mask=False, alpha1 = 1., alpha2 = 2., power = 1., info=None):
    '''
    states: ... x H x W
    Goal: HxW
    return: vector with size N
    '''
    original_shape = states.shape
    # batch_size = states.shape[0]
    # mask = torch.clamp(goal_sdf,0,0.1)*10
    if soft_mask:
        threshold1 = 0.0
        threshold2 = 0.1
        mask = 1-(torch.clamp(goal_sdf,threshold1,threshold2)-threshold1)*1/(threshold2-threshold1)
    else:
        mask = torch.zeros_like(goal_sdf)
        mask[torch.where(goal_sdf==0)]=1.
    loss = alpha1*(states * goal_sdf ** power).view(-1, original_shape[-1]*original_shape[-2]).sum(axis=1) - alpha2*(states * mask).view(-1, original_shape[-1]*original_shape[-2]).sum(axis=1)
    # loss -= 
    return loss.view(original_shape[:-2])


def gen_pose_list_batch(poses, horizon):
    '''
    interpolate trajectory for line agent

    poses: Nx2x2 torch tensor
        [
            [[pos_start_x, pos_start_y], [pos_end_x, pos_end_y]],
            ...
        ]
    return:
        N x T x 7 torch tensor
        [
            [q1, q2, q3, q0, x, y, z]
        ]
    '''
    N = poses.shape[0]
    x0 = poses[:,0, 0]
    y0 = poses[:,0, 1]
    x1 = poses[:,1, 0]
    y1 = poses[:,1, 1]
    z = torch.zeros_like(x0)+Z0
    q1 = torch.zeros_like(x0)
    q2 = torch.zeros_like(x0)
    theta = torch.atan2(y1-y0,x1-x0)
    
    q0 = torch.cos(theta/2)
    q3 = torch.sin(theta/2)

    start = torch.vstack([q1, q2, q3, q0, x0,y0,z]).T.view(N,1,7)
    # import pdb; pdb.set_trace()
    end = torch.vstack([q1, q2, q3, q0, x1,y1,z]).T.view(N,1,7)

    # output = torch.zeros(N,horizon+1,7).to(poses)

    start_w = torch.linspace(1, 0, steps=horizon+1).view(1,-1,1).to(poses)
    end_w = torch.linspace(0, 1, steps=horizon+1).view(1,-1,1).to(poses)

    output = start * start_w +end * end_w
    return output.contiguous()