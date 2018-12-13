import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from rtpose import rtpose_model
import cv2
import numpy as np
import os

from scipy import ndimage

from time import time

in_size=(368,368)
limb_set = [(0,1), (0,2), (0,3), (2,4), (3,5), (1,6), (1,7), (6,8), (7,9), (8,10), (9,11), (1,12), (1,13), (12,14), (13,15), (14,16), (15,17)]
colors = [(255,0,0), (255,128,0), (255,255,0), (255,255,0), (0,255,0), (0,255,0), (0,255,255), (0,255,255), (0,0,255), (0,0,255), (127,0,255), (127,0,255), 
(127,0,255), (127,0,255), (127,0,255), (127,0,255), (127,0,255), (127,0,255)]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = rtpose_model()

if os.path.exists('rtpose.pt'):
        model.load_state_dict(torch.load('rtpose.pt'))

model = model.to(device)

model.eval()

with torch.no_grad():
    # curr_img = cv2.imread('samples/test1.jpeg')
    # curr_img = cv2.imread('samples/test2.jpeg')
    # curr_img = cv2.imread('samples/test3.jpeg')
    curr_img = cv2.imread('samples/test4.jpeg')
    # curr_img = cv2.imread('samples/test5.jpeg')

    max_side = max(curr_img.shape[:2])

    y_pad = int((max_side - curr_img.shape[0])/2)
    x_pad = int((max_side - curr_img.shape[1])/2)

    curr_img = np.pad(curr_img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)

    x_orig = curr_img.shape[1]
    y_orig = curr_img.shape[0]

    curr_img = cv2.resize(curr_img, in_size)

    curr_img_np = curr_img.copy()

    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    curr_img = curr_img.astype(np.float32)
    curr_img = curr_img/255.0

    curr_img = torch.from_numpy(curr_img.transpose(2,0,1)).unsqueeze(0)
    curr_img = curr_img.to(device)

    last_layer, _ = model(curr_img)

    kps, pafs = last_layer[0][0], last_layer[1][0]
    kps = kps.cpu().data.numpy()
    pafs = pafs.cpu().data.numpy()

    
    scale_x = curr_img.size(3)/kps.shape[2]
    scale_y = curr_img.size(2)/kps.shape[1]

    centers = []

    # cv2.imwrite('orig.png', curr_img_np)
    for i, kp in enumerate(kps):
        kp[kp < 0.3] = 0
        lab_img, num_labs = ndimage.label(kp)
        # _, _, _, curr_centers = ndimage.measurements.extrema(kp, lab_img, list(range(1, num_labs + 1)))
        curr_centers = ndimage.measurements.center_of_mass(kp, lab_img, list(range(1, num_labs + 1)))

        # if i == 0:
        #     print(curr_centers)
        # print(curr_centers)
        centers.append([( int(round(center[1] * scale_x)), int(round(center[0] * scale_y)) ) for center in curr_centers])

        for c in centers[-1]:
            cv2.circle(curr_img_np, c, 2, colors[i], -1)
        # if i == 0:
        #     cv2.imwrite('kp_' + str(i) + '.png', kp * 255)
    
    cv2.imwrite('kp_draw.png', curr_img_np)
    

    # print(centers)