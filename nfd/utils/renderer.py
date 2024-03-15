import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc

def render_field(pose, field, points):
    '''
        pose: N(batch size x time step)x7 torch array
        return: 
            Nx1xHxW
    '''
    field, bounds_field= field['data'].to(pose), field['bounds']

    batch_size = pose.shape[0]
    pos = pose[:,4:-1]
    rot = pose[:,:4]
    q0, q1, q2, q3 = rot[:,3], rot[:,0], rot[:,1], rot[:,2]
    # H, W = self.cam_config['image_size']
    # height, width = self.in_shape[0], self.in_shape[1]
    # pix_size = (self.bounds[0,1] - self.bounds[0,0])/width

    points = torch.tensor(points).to(pose)
    points = points.repeat(batch_size, 1,1,1)
    
    rot_matrix = torch.zeros((batch_size, 2,2)).to(pose)
    rot_matrix[:,0,0] = 1 - 2*(q2**2+q3**2)
    rot_matrix[:,0,1] = 2*(q1*q2-q0*q3)
    rot_matrix[:,1,0] = 2*(q1*q2 +q0*q3)
    rot_matrix[:,1,1] = 1 - 2*(q1**2+q3**2)

    # print(points.dtype, pos.dtype, rot_matrix.dtype)
    # points = points-pos.view(batch_size,1, 1,2)
    points -= pos.view(batch_size,1, 1,2)
    # print(points.shape)
    # points = (points-pos) @ rot_matrix
    points = torch.einsum('baij,bjk->baik', points, rot_matrix)
    # torch.cuda.empty_cache(); gc.collect()
    center = (bounds_field[1] + bounds_field[0])/2

    # point in range [-1, 1]
    # points = (points-center)/(bounds_field[1] - center)
    points -= center
    points /= bounds_field[1] - center

    # input()
    # output = F.grid_sample(field[None, None, ...], points)
    output = F.grid_sample(field.repeat(batch_size,1, 1,1), points, align_corners=False)

    return output

def test_render_field():

    PUSHER_FIELD = 'ur5/spatula/spatula-rasterization.npy'
    ZONE_FIELD = 'zone/zone-rasterization.npy'
    ZONE_FIELD_SDF = 'zone/zone-rasterization-sdf.npy'

    assets_root = 'cliport/environments/assets/'

    pusher_field = np.load(os.path.join(assets_root, PUSHER_FIELD),allow_pickle=True).item()
    pusher_field['data'] = torch.tensor(pusher_field['data'].astype(np.float32))

    in_shape = (320, 160, 6)
    height, width = in_shape[0], in_shape[1]

    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    pix_size = (bounds[0,1] - bounds[0,0])/width
    
    
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    # print(px.shape)
    # print(py.shape)
    px = pix_size*px+bounds[0,0]
    py = pix_size*py+bounds[1,0]

    points = np.concatenate([px[...,None], py[...,None]], axis=2)

    # waypoints = sample_waypoints(N=128, bounds=self.bounds)
    waypoints = np.array([0.3874, -0.1611,  0.5,  0.0]).reshape((-1,4))
    waypoints = torch.tensor(waypoints, dtype=torch.float)
    sampled_trajs = gen_pose_list_batch(waypoints, horizon=10)
    print(sampled_trajs.shape)
    tic = time.time()
    for i in range(1000):
        actions = render_field(sampled_trajs[0,...], pusher_field, points)
    toc=time.time()
    print(toc-tic)
    print(actions.shape)

    plt.figure()
    plt.imshow(actions[0,0,...].numpy())
    plt.show()

# @torch.jit.script
def render_field2(pose, field, bounds_field, points):
    '''
        pose: N(batch size x time step)x7 torch array
        return: 
            Nx1xHxW
    '''
    # field, bounds_field= field['data'].to(pose), field['bounds']

    batch_size = pose.shape[0]
    pos = pose[:,4:-1]
    rot = pose[:,:4]
    q0, q1, q2, q3 = rot[:,3], rot[:,0], rot[:,1], rot[:,2]
    # H, W = self.cam_config['image_size']
    # height, width = self.in_shape[0], self.in_shape[1]
    # pix_size = (self.bounds[0,1] - self.bounds[0,0])/width

    # points = torch.tensor(points).to(pose)
    points = points.repeat(batch_size, 1,1,1)
    
    rot_matrix = torch.zeros((batch_size, 2,2)).to(pose)
    rot_matrix[:,0,0] = 1 - 2*(q2**2+q3**2)
    rot_matrix[:,0,1] = 2*(q1*q2-q0*q3)
    rot_matrix[:,1,0] = 2*(q1*q2 +q0*q3)
    rot_matrix[:,1,1] = 1 - 2*(q1**2+q3**2)

    # print(points.dtype, pos.dtype, rot_matrix.dtype)
    # points = points-pos.view(batch_size,1, 1,2)
    points -= pos.view(batch_size,1, 1,2)
    # print(points.shape)
    # points = (points-pos) @ rot_matrix
    points = torch.einsum('baij,bjk->baik', points, rot_matrix)
    # torch.cuda.empty_cache(); gc.collect()
    center = (bounds_field[1] + bounds_field[0])/2

    # point in range [-1, 1]
    # points = (points-center)/(bounds_field[1] - center)
    points -= center
    points /= bounds_field[1] - center

    # input()
    # output = F.grid_sample(field[None, None, ...], points)
    output = F.grid_sample(field.repeat(batch_size,1, 1,1), points, align_corners=False)

    return output


# render_field2 = torch.jit.trace(render_field2, (torch.rand(20,7), torch.rand(40,40), torch.rand(2), torch.rand(320, 160, 2)))

# @torch.jit.script
# def test_render_field3(pose: torch.Tensor, field: torch.Tensor, bounds_field: torch.Tensor, points: torch.Tensor):
#     for i in range(15):
#         render_field2(pose, field, bounds_field, points)
#     pass
#     return render_field2(pose, field, bounds_field, points)
# print(type(test_render_field3))
# render_field3 = torch.jit.trace(test_render_field3, (torch.rand(20,7), torch.rand(40,40), torch.rand(2), torch.rand(320, 160, 2)))

def test_render_field_jit():
    

    PUSHER_FIELD = 'ur5/spatula/spatula-rasterization.npy'
    ZONE_FIELD = 'zone/zone-rasterization.npy'
    ZONE_FIELD_SDF = 'zone/zone-rasterization-sdf.npy'

    assets_root = 'cliport/environments/assets/'

    pusher_field = np.load(os.path.join(assets_root, PUSHER_FIELD),allow_pickle=True).item()
    
    pusher_field_bounds = torch.tensor(pusher_field['bounds'])
    pusher_field = torch.tensor(pusher_field['data'].astype(np.float32))

    in_shape = (320, 160, 6)
    height, width = in_shape[0], in_shape[1]

    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    pix_size = (bounds[0,1] - bounds[0,0])/width
    
    
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    # print(px.shape)
    # print(py.shape)
    px = pix_size*px+bounds[0,0]
    py = pix_size*py+bounds[1,0]

    points = np.concatenate([px[...,None], py[...,None]], axis=2)
    
    tic = time.time()

    waypoints = sample_waypoints(N=128, bounds=bounds)
    toc=time.time()
    print(sample_waypoints, toc-tic)
    # waypoints = np.array([0.3874, -0.1611,  0.5,  0.0]).reshape((-1,4))
    waypoints = torch.tensor(waypoints, dtype=torch.float)
    sampled_trajs = gen_pose_list_batch(waypoints, horizon=16)
    sampled_trajs = sampled_trajs.reshape(-1, sampled_trajs.shape[-2], sampled_trajs.shape[-1])
    print(sampled_trajs.shape)

    points = torch.tensor(points).to(sampled_trajs)
    pusher_field.to(sampled_trajs)
    pusher_field_bounds.to(sampled_trajs)
    # render_field2 = torch.jit.trace(render_field2, (sampled_trajs[0,...], pusher_field, pusher_field_bounds, points))
    
    print(sampled_trajs[0,...].shape, 'traj')
    print(pusher_field.shape, 'pusher_field')
    print(pusher_field_bounds, 'bounds')
    print(points.shape, 'points')
    tic = time.time()
    for i in range(16):
        actions = render_field2(sampled_trajs[0,...], pusher_field, pusher_field_bounds, points)
    toc=time.time()
    print('field 2', toc-tic)
    tic = time.time()
    render_field3(sampled_trajs[0,...], pusher_field, pusher_field_bounds, points)
    toc=time.time()
    print('field 3', toc-tic)
    # actions = render_field2(sampled_trajs[0,...], pusher_field, pusher_field_bounds, points)
    print(actions.shape)

    plt.figure()
    plt.imshow(actions[0,0,...].numpy())
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    from nfd.agents.ndf_agent import sample_waypoints, gen_pose_list_batch
    import time

    # test_render_field()
    test_render_field_jit()