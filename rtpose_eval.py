import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rtpose import rtpose_model
import cv2
from PIL import Image
import numpy as np
import os

from scipy import ndimage

from time import time

from scipy.ndimage.filters import gaussian_filter

import utils

in_size=(368,368)
limb_set = [(0,1), (0,2), (0,3), (2,4), (3,5), (1,6), (1,7), (6,8), (7,9), (8,10), (9,11), (1,12), (1,13), (12,14), (13,15), (14,16), (15,17)]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

'''
        Keypoints COCO (augemented w/ neck):
        0:nose	   	    1: neck         2: l eye        3: r eye	 4: l ear	  5: r ear        
        6: l shoulder	7: r shoulder	8: l elbow	    9: r elbow  10: l wrist  11: r wrist
        12: l hip	   13: r hip       14: l knee      15: r knee   16: l ankle	 17: r ankle
'''
def return_prediction(img, viz=False, path=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = rtpose_model()

    if os.path.exists('rtpose.pt'):
        model.load_state_dict(torch.load('rtpose.pt'))

    model = model.to(device)

    model.eval()

    with torch.no_grad():

        if path:
            img = cv2.imread(img)
            orig_img = img.copy()

        orig_y, orig_x = img.shape[:2]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0

        img, (xp, yp), (xo, yo) = utils.resize_and_pad_img(img, in_size)
        img = img.to(device)

        last_layer, _ = model(img)

        kps, pafs = last_layer
        kps = kps[0].data.cpu().numpy()
        pafs = pafs[0].data.cpu().numpy()

        peaks = utils.get_kps(kps, in_size)

        inv_scale_x = xo/img.size(2)
        inv_scale_y = yo/img.size(3)

        peaks_adj = []
        for i in range(18):
            peaks_adj.append([])

        for i, peaks in enumerate(peaks):
            for pt in peaks:
                x = int((pt[0] * inv_scale_x) - xp)
                y = int((pt[1] * inv_scale_y) - yp)
                orig_img = cv2.circle(orig_img, (x,y), 3, colors[i], thickness=-1)

                peaks_adj[i].append((x, y, pt[2], pt[3]))
                
        cv2.imwrite('viz_img.png', orig_img)

    return peaks_adj


if __name__ == "__main__":
    # curr_img = 'samples/test1.jpeg'
    # curr_img = 'samples/test2.jpeg'
    # curr_img = 'samples/test3.jpeg'
    # curr_img = 'samples/test4.jpeg'
    curr_img = 'samples/test5.jpeg'
    # curr_img = 'samples/test6.png'

    return_prediction(curr_img)
