import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from rtpose import rtpose_model
import cv2
from PIL import Image
import numpy as np
import os

from scipy import ndimage

from time import time

from scipy.ndimage.filters import gaussian_filter

in_size=(368,368)
limb_set = [(0,1), (0,2), (0,3), (2,4), (3,5), (1,6), (1,7), (6,8), (7,9), (8,10), (9,11), (1,12), (1,13), (12,14), (13,15), (14,16), (15,17)]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def resize_and_pad_img(img, method='torch', viz=False):
    orig_img = None
    if method == 'numpy':

        max_side = max(img.shape[:2])

        y_pad = int((max_side - img.shape[0])/2)
        x_pad = int((max_side - img.shape[1])/2)

        img = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)

        img = cv2.resize(img, in_size)

        orig_img = None
        if viz:
            orig_img = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0

        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    
    elif method == 'torch':
        orig_img = None
        if viz:
            orig_img = img.copy()

        max_side = max(img.size)

        y_pad = int((max_side - img.size[0])/2)
        x_pad = int((max_side - img.size[1])/2)

        padder = transforms.Pad((x_pad, y_pad))

        img = padder(img)

        resizer = transforms.Resize(in_size)

        img = resizer(img)

        to_tensor = transforms.ToTensor()

        img = to_tensor(img)
        img = img.unsqueeze(0)

    
    return img, orig_img

'''
        Keypoints COCO (augemented w/ neck):
        0:nose	   	    1: neck         2: l eye        3: r eye	 4: l ear	  5: r ear        
        6: l shoulder	7: r shoulder	8: l elbow	    9: r elbow  10: l wrist  11: r wrist
        12: l hip	   13: r hip       14: l knee      15: r knee   16: l ankle	 17: r ankle
'''
def return_prediction(img, viz=False):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = rtpose_model()

    if os.path.exists('rtpose.pt'):
        model.load_state_dict(torch.load('rtpose.pt'))

    model = model.to(device)

    model.eval()

    with torch.no_grad():

        img = cv2.imread(img)

        img, orig_img = resize_and_pad_img(img, method='numpy', viz=True)
        img = img.to(device)

        # cv2.imwrite('out.png', orig_img)

        last_layer, _ = model(img)

        kps, pafs = last_layer[0][0], last_layer[1][0]
        kps = kps.cpu().data.numpy()
        pafs = pafs.cpu().data.numpy()

        # Taken from the Openpose Python Code 
        all_peaks = []
        peak_counter = 0

        for i in range(kps.shape[0]):
            x_list = []
            y_list = []

            curr_kp = cv2.resize(kps[i], (img.size(3), img.size(2)), interpolation=cv2.INTER_CUBIC)
            g_map = gaussian_filter(curr_kp, sigma=3)
            
            map_left = np.zeros(g_map.shape)
            map_left[1:,:] = g_map[:-1,:]
            map_right = np.zeros(g_map.shape)
            map_right[:-1,:] = g_map[1:,:]
            map_up = np.zeros(g_map.shape)
            map_up[:,1:] = g_map[:,:-1]
            map_down = np.zeros(g_map.shape)
            map_down[:,:-1] = g_map[:,1:]

            peaks_binary = np.logical_and.reduce((g_map>=map_left, g_map>=map_right, g_map>=map_up, g_map>=map_down, g_map > 0.6))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
            peaks_with_score = [x + (curr_kp[x[1],x[0]],) for x in peaks]
            pt_type = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (pt_type[i],) for i in range(len(pt_type))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        for i, peaks in enumerate(all_peaks):
            for pt in peaks:
                orig_img = cv2.circle(orig_img, pt[:2], 4, colors[i], thickness=-1)

        cv2.imwrite('viz_kp.png', orig_img)
        
        
        # cv2.imwrite('kp_draw.png', orig_img)


if __name__ == "__main__":
    # curr_img = 'samples/test1.jpeg'
    # curr_img = 'samples/test2.jpeg'
    # curr_img = 'samples/test3.jpeg'
    # curr_img = 'samples/test4.jpeg'
    curr_img = 'samples/test5.jpeg'
    # curr_img = 'samples/test6.png'

    return_prediction(curr_img)
