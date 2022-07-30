import math
import numpy as np
import cv2
import os
import yaml


with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

path_to_fig = config['testing']['fig_path']

def taylor(hm, coord):
    heatmap_width = hm.shape[0]
    px = int(coord)
    
    if 1 < px < heatmap_width-2 :

        dx  = 0.5 * (hm[px+1] - hm[px-1])
        dxx = 0.25 * (hm[px+2] - 2 * hm[px] + hm[px-2])
        derivative = np.matrix([dx])
        hessian = np.matrix([dxx])
        if dxx ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset

    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    # batch_size = hm.shape[0]
    num_joints = 6
    # height = hm.shape[2]
    width = hm.shape[0]
    origin_max = np.max(hm)
    dr = np.zeros((width + 2 * border))
    dr[border: -border] = hm.copy()
    kernel_1d = cv2.getGaussianKernel(kernel,2)
    dr = cv2.filter2D(dr, -1, kernel_1d)[:,0]
    hm = dr[border: -border].copy()
    hm *= origin_max / np.max(hm)


    return hm

def save_as_img(prop, vid_name, branch):
    prop = prop*255
    
    if branch =="bottom":
        save_path = os.path.join(path_to_fig,"GSM_PIC_BOTTOM")
        cv2.imwrite(os.path.join(save_path,vid_name+".png"), prop)
    else:
        save_path = os.path.join(path_to_fig,"GSM_PIC_TOP")
        cv2.imwrite(os.path.join(save_path,vid_name+".png"), prop)


def save_as_img_feat(im, layer):
    im = im*255
    im = np.stack((im,)*3, axis=-1).astype(np.uint8)
    im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    
    save_path = os.path.join(path_to_fig,"GSM_PIC_FEAT")
    if layer=="before":

        cv2.imwrite(os.path.join(save_path,"FEAT_BEFORE_.png"), im)
    else:
        cv2.imwrite(os.path.join(save_path,"FEAT_AFTER_.png"), im)
    

    
