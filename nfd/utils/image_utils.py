import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from PIL import Image
import subprocess
import os
import IPython

import random
import string

import re
import io

try:
    from scipy import signal
    import cv2
    from skimage import io
    from scipy.spatial.transform import Rotation as R
    import torch
except:
    pass

import tempfile

import matplotlib as mpl

'''Useful sctipts:
mpl.rcParams['figure.dpi']= 400
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from matplotlib.animation import FuncAnimation
def animate_fun(frame_num): pass
anim = FuncAnimation(fig, animate_fun, frames=100, interval=20)

%matplotlib inline
'''


def set_dpi(dpi=400):
    mpl.rcParams['figure.dpi']= 400

def show_img(img):
    if img.max()<=1:
        img = Image.fromarray((img * 255).astype(np.uint8))#.save(f,'png')
    else:
        img=  Image.fromarray((img).astype(np.uint8))
#     import IPython
    IPython.display.display(img)


def crop_img(img,center,factor):
    '''
    keep shape crop
    '''
    res = img.shape[:2]
#     print(res)
    start_h, end_h = int(res[0]*(-factor/2+center[0])),int(res[0]*(factor/2+center[0]))
    start_w, end_w = int(res[1]*(-factor/2+center[1])),int(res[1]*(factor/2+center[1]))
#     print(start_h, end_h)
#     print(start_w, end_w)
    img = cv2.resize(img[start_h:end_h+1,start_w:end_w+1,:],res[::-1])
    return img

def stack_img(img_mat,sep,gap_rgb=[255,255,255]):
    '''
    stack img to matrix
    img_mat:[[img1,img2],[img3,img4]]
    sep: (gap_size_height, gap_size_weight)
    '''
    if img_mat[0][0].max()<=1:
        gap_rgb=np.array(gap_rgb)/255.
    
    img_lines = []
    for j,line in enumerate(img_mat):
        line_new = []
        for i in range(len(line)):
            if i==0:
                w_gap_h = line[i].shape[0]
                w_gap = np.ones((w_gap_h,sep[1],3))
                w_gap[...,:] = np.array(gap_rgb)
                w_gap = w_gap.astype(line[i].dtype)
                line_new.append(line[i])
            else:
                line_new.append(w_gap)
                line_new.append(line[i])
        # print([s.shape for s in line_new])
        img_line = cv2.hconcat(line_new)
        if j==0:
            h_gap_w = img_line.shape[1]
            h_gap = np.ones((sep[0],h_gap_w,3))
            h_gap[...,:] = np.array(gap_rgb)
            h_gap = h_gap.astype(img_line.dtype)
        else:
            img_lines.append(h_gap)
        img_lines.append(img_line)
    img_mat = cv2.vconcat(img_lines)
    return img_mat

def transform_points(tr, pts):
    """
    homogenuous points transformation
    pts NxD or Nx(D-1) 
    tr DxD
    >>> camera_matrix = torch.eye(3)
    >>> camera_matrix[1,2] = -100 # crop
    >>> camera_matrix[0,0] = 2 # scale
    >>> camera_pts = torch.rand(100,3)*200
    >>> warp_pts = transform_points_th(camera_matrix, camera_pts)
    """
    # homogenuous
    pts = torch.Tensor(pts)
    tr = torch.Tensor(tr)
    if  pts.shape[-1]+1 == tr.shape[-1]:
        pts = torch.cat([pts, torch.ones_like(pts[...,0:1])],dim=-1) 
    pts_hom_tr = pts @ (tr.T)
    pts_tr = pts_hom_tr[..., :-1] / pts_hom_tr[..., -1,None]
    return pts_tr


class GIFSaver(object):
    """docstring for GIFSaver"""
    def __init__(self, name=None, path=None, temp_format="png"):
        super(GIFSaver, self).__init__()
        # self.arg = arg
        # self.count=0
        if name is not None:
            self.name = name
            # self.isname = True
        else:
            # self.isname = False
            self.name = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
        self.path = path
        self.temp_format = temp_format.lower()
        self.temp_path = tempfile.gettempdir()
        self.file_list = []
        self.fig_list = []

    def __call__(self,count):
        fname = 'gif_tmp_'+self.name+f'_{count}.{self.temp_format}'
        fpath = os.path.join(self.temp_path, fname)
        self.file_list.append(fpath)
        return fpath
    
    def add(self, img):
        if img.max()>1:
            img = Image.fromarray(np.uint8(img))
        else:
            img = Image.fromarray(np.uint8(img*255))
        self.fig_list.append(img)

    def save(self,name=None,path=None, duration=500, loop=0):
        if name :
            if os.sep in name:
                output_path = name
            elif path is not None:
                output_path = os.path.join(path, name)
            elif self.path is not None:
                output_path = os.path.join(self.path, name)
            else:
                output_path = os.path.join(os.getcwd(), name)
        else:
            if path is not None:
                output_path = os.path.join(path, self.name)
            elif self.path is not None:
                output_path = os.path.join(self.path, self.name)
            else:
                output_path = os.path.join(os.getcwd(), self.name)
        if not output_path.endswith('.gif'):
            output_path+='.gif'

        if not self.fig_list:
            images=[]
            for img_file in self.file_list:
                im = Image.open(img_file)
                images.append(im)
        else:
            images = self.fig_list
        images[0].save(os.path.join(output_path), save_all=True, append_images=images[1:], duration=duration, loop=loop)
        for img_file in self.file_list:
            os.remove(img_file)

if __name__ == '__main__':
    def test_GIFSaver():

        a = GIFSaver()

        fig = plt.figure()
        x = np.arange(10)
        for i in range(10):
            plt.gca().clear()
            y = np.sin(x+i)
            plt.plot(x,y)
            plt.pause(0.1)
            fig.savefig(a(i))

        a.save("test")
        plt.show()
    def test_GIFSaver2():

        a = GIFSaver()

        a.add(np.ones((10,10,3)))
        a.add(np.zeros((10,10,3)))
        a.add(np.ones((10,10,3)))
        a.add(np.zeros((10,10,3)))

        a.save("test2")
    test_GIFSaver2()



        