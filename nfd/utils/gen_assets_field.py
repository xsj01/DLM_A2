import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skfmm

assets_root = 'cliport/environments/assets/'
PUSHER_FIELD = 'ur5/spatula/spatula-rasterization.npy'
# PUSHER_FIELD = 'ur5/spatula/spatula-long-rasterization.npy'
# PUSHER_FIELD = 'ur5/spatula/spatula-short-rasterization.npy'
ZONE_FIELD = 'zone/zone-rasterization.npy'
ZONE_FIELD_SDF = 'zone/zone-rasterization-sdf.npy'

def gen_pusher_rasterization():
    # H,W = 80,80
    H,W = 40,40
    data = np.zeros((H,W))
    dx = 20
    # dx = 40
    # dx = 10
    dy = 2

    start_h = (H-dy)//2
    start_w = (W-dx)//2

    data[start_h:start_h+dy, start_w:start_w+dx] = 1.
    # data[19:21, 10:30] = 1.
    bounds = [-0.05, 0.05]
    # bounds = [-0.1, 0.1]
    plt.figure()
    plt.imshow(data)
    plt.show()
    data = {'data':data.T, 'bounds':bounds}
    # with open
    np.save(os.path.join(assets_root, PUSHER_FIELD), data)
    pass

def gen_zone_rasterization():
    H, W = 100,100
    data = np.zeros((H,W))

    dx = 12
    dy = 12

    start_h = (100-dy)//2
    start_w = (100-dx)//2

    data[start_h:start_h+dy, start_w:start_w+dx] = 1.
    bounds = [-0.5, 0.5]
    plt.figure()
    plt.imshow(data)
    plt.show()
    data = {'data':data, 'bounds':bounds}
    # with open
    np.save(os.path.join(assets_root, ZONE_FIELD), data)

def gen_zone_sdf():
    H, W = 200,200
    data = np.ones((H,W))

    dx = 12
    dy = 12

    start_h = (H-dy)//2
    start_w = (W-dx)//2
    bounds = [-1., 1.]

    data[start_h:start_h+dy, start_w:start_w+dx] = -1.
    bounds = [-1, 1.]
    data = skfmm.distance(data,dx=0.01)
    data = np.clip(data, 0, np.inf)
    # print(len(np.where(data==0)[0]))
    # data[]=100
    plt.figure()
    plt.imshow(data)
    plt.show()

    data = {'data':data, 'bounds':bounds}

    np.save(os.path.join(assets_root, ZONE_FIELD_SDF), data)


if __name__ == '__main__':
    gen_pusher_rasterization()
    # gen_zone_rasterization()
    # gen_zone_sdf()