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
limb_set = [(0,1), (0,2), (0,3), (2,4), (3,5), (1,6), (1,7), (6,8), (7,9), (8,10), (9,11), (1,12), (1,13), (12,14), (13,15), (14,16), (15,17)]

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

def pafOnPt(paf_total, point1, point2, sigma):
    paf_vec = point2 - point1
    paf_tmp = np.zeros(paf_total.shape)

    limb_norm = np.sqrt(np.sum(np.power(paf_vec, 2), axis=0))

    if limb_norm != 0:
        v = paf_vec/limb_norm
        v_perp = np.array([v[1], -v[0]])

        x1 = 0
        x2 = paf_total.shape[1]
        y1 = 0
        y2 = paf_total.shape[0]
        grid_x = np.linspace(x1, x2 - 1, x2 - x1)
        grid_y = np.linspace(y1, y2 - 1, y2 - y1)

        x_vals, y_vals = np.meshgrid(grid_x, grid_y)

        x_vals = x_vals - point1[0]
        y_vals = y_vals - point1[1]

        x_vals_1 = x_vals * v[0]
        y_vals_1 = y_vals * v[1]
        total_v = x_vals_1 + y_vals_1

        x_vals_2 = x_vals * v_perp[0]
        y_vals_2 = y_vals * v_perp[1]
        total_v_perp = np.abs(x_vals_2 + y_vals_2)
        logical_idx = (total_v >= 0) & (total_v <= limb_norm) & (total_v_perp < sigma)

        paf_tmp[:,:,0][logical_idx] = v[0]
        paf_tmp[:,:,1][logical_idx] = v[1]

        paf_total[:,:,0] += paf_tmp[:,:,0]
        paf_total[:,:,1] += paf_tmp[:,:,1]
        
    return paf_total

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
    
    '''
        Keypoints COCO (augemented w/ neck):
        0: nose	   		1: neck         2: l eye    3: r eye	4: l ear	  5: r ear
        6: l shoulder	7: r shoulder	8: l elbow	9: r elbow  10: l wrist	 11: r wrist		
        12: l hip	   13: r hip       14: l knee  15: r knee   16: l ankle	 17: r ankle
    '''
    def __getitem__(self, idx):
        img_path, ann_path = self.data[idx]
        print(img_path)
        img = io.imread(img_path)
        curr_json_f = open(ann_path, 'r')
        anns = json.load(curr_json_f)
        
        intermediate_reprs = []
        for ann in anns:
            curr_kp_repr = {}
            for i in range(0, len(ann['keypoints']), 3):
                kp = i/3
                if ann['keypoints'][i+2] != 0:
                    curr_kp_repr[kp] = np.array(ann['keypoints'][i:i+2])
                else:
                    curr_kp_repr[kp] = None
            intermediate_reprs.append(curr_kp_repr)

            for k in sorted(curr_kp_repr.keys(), reverse=True):
                if i/3 > 0:
                    curr_kp_repr[i/3+1] = curr_kp_repr[i/3]
            
            if curr_kp_repr[6] is not None and curr_kp_repr[7] is not None:
                curr_kp_repr[1] = (curr_kp_repr[6] + curr_kp_repr[7])/2
                curr_kp_repr[1] = curr_kp_repr[1].astype(np.int)
            else:
                curr_kp_repr[1] = None

        kp_maps = {}
        for kps in intermediate_reprs:
            for i in kps.keys():
                if i not in kp_maps.keys():
                    kp_maps[i] = np.zeros((img.shape[0], img.shape[1]))
                if kps[i] is not None:
                    kp_maps[i] = gaussianOnPt(kp_maps[i], kps[i], 1)
        
        paf_maps = {}
        paf_counts = {}
        for kps in intermediate_reprs:
            for limb in limb_set:
                if limb not in paf_maps.keys():
                    paf_maps[limb] = np.zeros((img.shape[0], img.shape[1], 2))
                    paf_counts[limb] = 0
                f,t = limb
                pt1 = kps[f]
                pt2 = kps[t]
                if pt1 is not None and pt2 is not None:
                    paf_maps[limb] = pafOnPt(paf_maps[limb], pt1, pt2, 2)
                    paf_counts[limb] += 1
        
        curr_json_f.close()
        return (img, kp_maps, paf_maps)

base_path = '/home/shared/workspace/coco_keypoints'
cocodset = CocoPoseDataset(os.path.join(base_path, 'images'), os.path.join(base_path, 'annotations'))

img, kp, paf = cocodset[1]