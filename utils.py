import torch
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

def resize_and_pad_img(img, size, viz=False):

    max_side = max(img.shape)

    y_pad = int((max_side - img.shape[0])/2)
    x_pad = int((max_side - img.shape[1])/2)

    img = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)

    orig_x, orig_y = img.shape[:2]

    img = cv2.resize(img, size)

    img = torch.from_numpy(img.transpose(2,0,1))
    img = img.unsqueeze(0)
    
    return img, (x_pad, y_pad), (orig_x, orig_y)

# Adapted from Openpose python code.
def get_kps(pred_kps, pad_img_size):

    all_peaks = []
    peak_counter = 0

    for i in range(pred_kps.shape[0]):
        curr_kp = cv2.resize(pred_kps[i], pad_img_size, interpolation=cv2.INTER_CUBIC)
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

    return all_peaks


