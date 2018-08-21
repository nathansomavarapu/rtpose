import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import glob
import os
import json

img_ext = ''
ann_ext = '.json'

def gaussianOnPt(conf_map_total, point, sigma):
    tmp_map = np.zeros((conf_map_total.shape[0], conf_map_total.shape[1]))
    gridx = np.linspace(-int(sigma * 2), int(sigma * 2), sigma * 4 + 1)
    gridy = np.linspace(-int(sigma * 2), int(sigma * 2), sigma * 4 + 1)
    x_vals, y_vals = np.meshgrid(gridx, gridy)
    
    grid = np.exp(- (np.sqrt(np.power(x_vals, 2) + np.power(y_vals, 2)))/ np.power(sigma, 2))
    
    tmp_map[point[1]-int(sigma * 2):point[1]+int(sigma * 2)+1, point[0]-int(sigma * 2):point[0]+int(sigma * 2)+1] = grid

    conf_map_total = np.maximum(conf_map_total, tmp_map)
    conf_map_total[conf_map_total > 1] = 1

    return conf_map_total

class CocoPoseDataset(Dataset):

    def __init__(self, img_dir, ann_dir, transforms=None):
        kps = glob.glob(os.path.join(ann_dir, '*' + ann_ext))
        imgs = glob.glob(os.path.join(img_dir, '*' + img_ext))
        self.data = []
        self.transforms = transforms

        print('Loading dataset paths...')
        for img in imgs:
            fname = img.split('/')[-1]
            ext = fname.split('.')[-1]
            kps_fname = os.path.join(ann_dir, fname.replace('.' + ext, ann_ext))
            
            if kps_fname in kps:
                self.data.append((img, kps_fname))
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        img_path, ann_path = self.data[idx]
        print(img_path)
        img = io.imread(img_path)
        curr_json_f = open(ann_path, 'r')
        anns = json.load(curr_json_f)
        
        kp_maps = {}
        for ann in anns:
            for i in range(0, len(ann['keypoints']), 3):
                if i not in kp_maps.keys():
                    kp_maps[i] = np.zeros((img.shape[0],img.shape[1]))
                if ann['keypoints'][i+2] != 0:
                    kp_maps[i] = gaussianOnPt(kp_maps[i], tuple(ann['keypoints'][i:i+2]), 1)
        
        curr_json_f.close()
        return (img, kp_maps)

base_path = '/home/shared/workspace/coco_keypoints'
cocodset = CocoPoseDataset(os.path.join(base_path, 'images'), os.path.join(base_path, 'annotations'))

print(cocodset[1])