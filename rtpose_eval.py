import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from rtpose import rtpose_model
import cv2
import numpy as np
import os

from scipy.ndimage.filters import maximum_filter

in_size=(368,368)
limb_set = [(0,1), (0,2), (0,3), (2,4), (3,5), (1,6), (1,7), (6,8), (7,9), (8,10), (9,11), (1,12), (1,13), (12,14), (13,15), (14,16), (15,17)]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = rtpose_model()

if os.path.exists('rtpose.pt'):
        model.load_state_dict(torch.load('rtpose.pt'))

model = model.to(device)

model.eval()

with torch.no_grad():
    curr_img = cv2.imread('test.jpeg')

    max_side = max(curr_img.shape[:2])

    y_pad = int((max_side - curr_img.shape[0])/2)
    x_pad = int((max_side - curr_img.shape[1])/2)

    curr_img = np.pad(curr_img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)

    x_orig = curr_img.shape[1]
    y_orig = curr_img.shape[0]

    curr_img = cv2.resize(curr_img, in_size)

    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    curr_img = curr_img.astype(np.float32)
    curr_img = curr_img/255.0

    scale_x = curr_img.shape[1]/x_orig
    scale_y = curr_img.shape[0]/y_orig

    curr_img = torch.from_numpy(curr_img.transpose(2,0,1)).unsqueeze(0)
    curr_img = curr_img.to(device)
    last_layer, _ = model(curr_img)

    kps, pafs = last_layer[0][0], last_layer[1][0]
    kps = kps.cpu().data.numpy()
    pafs = pafs.cpu().data.numpy()

    for i, kp in enumerate(kps):
        cv2.imwrite('kp_' + str(i) + '.png', maximum_filter(kp, size=(3,3)) * 255)



    # fprint = generate_binary_structure(2,2)
    # for i in range(kp.shape[0]):
    #     local_max = maximum_filter(kp[i], footprint=fprint) == kp[i]

    #     background = (kp[i] == 0)

    #     eroded_background = binary_erosion(background, structure=fprint, border_value=1)

    #     detected_peaks = local_max ^ eroded_background

    #     # TODO: This doesnt do anything fix it.
    #     detected_peaks * (kp[i] > 1.9)

    #     print(np.array(np.nonzero(detected_peaks)))