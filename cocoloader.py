import torch
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import os
import json

import torch.nn.functional as F
from pycocotools.coco import COCO


img_ext = '.jpg'
ann_ext = '.json'
limb_set = [(0,1), (0,2), (0,3), (2,4), (3,5), (1,6), (1,7), (6,8), (7,9), (8,10), (9,11), (1,12), (1,13), (12,14), (13,15), (14,16), (15,17)]

def put_gaussian(point, gauss_acc, sigma, stride):
    start = stride / 2.0 - 0.5
    
    x_grid, y_grid = np.meshgrid(np.arange(gauss_acc.shape[0]), np.arange(gauss_acc.shape[1]))

    x_grid = x_grid * stride + start
    y_grid = y_grid * stride + start

    dist = (np.power((x_grid - point[0]), 2) +  np.power((y_grid - point[1]), 2))/(2.0 * np.power(sigma, 2))
    dist_temp = np.exp(-dist)
    dist_temp[dist >= 4.6052] = 0

    gauss_acc = np.max(np.dstack((gauss_acc, dist_temp)), axis=2)
    gauss_acc[gauss_acc > 1] = 1
    
    return gauss_acc

def put_paf(point1, point2, paf_acc, theta, stride):
    point1 = np.array([point1[0]/float(stride), point1[1]/float(stride)])
    point2 = np.array([point2[0]/float(stride), point2[1]/float(stride)])

    tmp_paf_0 = np.zeros((paf_acc.shape[0], paf_acc.shape[1]))
    tmp_paf_1 = np.zeros((paf_acc.shape[0], paf_acc.shape[1]))
    count_map = np.zeros((paf_acc.shape[0], paf_acc.shape[1]))

    v = point2 - point1
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return paf_acc, count_map

    v_unit = v/v_norm
    v_perp = np.array([v_unit[1], -v_unit[0]])

    x_grid, y_grid = np.meshgrid(np.arange(paf_acc.shape[0]), np.arange(paf_acc.shape[1]))

    dist_0 = v_unit[0] * (x_grid - point1[0]) + v_unit[1] * (y_grid - point1[1])
    dist_1 = np.abs(v_perp[0] * (x_grid - point1[0]) + v_perp[1] * (y_grid - point1[1]))

    tmp_paf_0[(dist_0 >= 0) & (dist_0 <= v_norm) & (dist_1 <= theta)] = v_unit[0]
    tmp_paf_1[(dist_0 >= 0) & (dist_0 <= v_norm) & (dist_1 <= theta)] = v_unit[1]
    count_map[(dist_0 >= 0) & (dist_0 <= v_norm) & (dist_1 <= theta)] = 0

    return paf_acc + np.dstack([tmp_paf_0, tmp_paf_1]), count_map

class CocoPoseDataset:

    def __init__(self, ann_dir, img_dir, size=(368, 368), end_size=(46,46), theta=1.0, sigma=7.0, stride=8):
        self.coco=COCO(ann_dir)
        self.imgs = self.coco.getImgIds()
        self.img_path = img_dir

        self.img_size = size
        self.end_size = end_size
        self.sigma = sigma
        self.theta = theta
        self.stride = stride

        ia.seed(np.random.randint(1000))

        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-180, 180)
            )
        ])
    
    def __len__(self):
        return len(self.imgs)

    '''
        Keypoints COCO (augemented w/ neck):
        0:nose	   	1: neck         2: l eye        3: r eye	 4: l ear	  5: r ear        
        6: l shoulder	7: r shoulder	8: l elbow	    9: r elbow  10: l wrist  11: r wrist
        12: l hip	   13: r hip       14: l knee      15: r knee   16: l ankle	 17: r ankle
    '''
    def __getitem__(self, index):

        curr_img_id = self.imgs[index]
        ann_ids = self.coco.getAnnIds(imgIds=curr_img_id)
        curr_ann = self.coco.loadAnns(ann_ids)
        img_f = os.path.join(self.img_path, self.coco.loadImgs(curr_img_id)[0]['file_name'])

        curr_img = np.array(cv2.imread(img_f), dtype=np.uint8)

        max_side = max(curr_img.shape[:2])

        y_pad = int((max_side - curr_img.shape[0])/2)
        x_pad = int((max_side - curr_img.shape[1])/2)

        # pad image to sqaure
        curr_img = np.pad(curr_img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)

        x_orig = curr_img.shape[1]
        y_orig = curr_img.shape[0]

        curr_img = cv2.resize(curr_img, self.img_size)

        aug_det = self.augmenter.to_deterministic()
        curr_img = aug_det.augment_image(curr_img)

        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        curr_img = curr_img.astype(np.float32)
        curr_img = curr_img/255.0

        # TODO: Mean center images

        curr_h, curr_w = curr_img.shape[:2]

        scale_x = curr_w/x_orig
        scale_y = curr_h/y_orig

        if len(curr_ann) != 0:
            ann_list = []
            for ann_idx in range(len(curr_ann)):
                kpts = curr_ann[ann_idx]['keypoints']

                point_dict = {}
                for i in range(0, len(kpts), 3):
                    kp = kpts[i:i+3]
                    kp_idx = i if i == 0 else (i//3)+1
                    if kp[2] > 0:
                        point_dict[kp_idx] = [round((kp[0] + x_pad) * scale_x), round((kp[1] + y_pad) * scale_y), kp[2]]
                    else:
                        point_dict[kp_idx] = []
                            
                ls = point_dict[6]
                rs = point_dict[7]

                if len(ls) != 0 and len(rs) != 0 and ls[2] > 0 and rs[2] > 0:
                        indicator = 2 if ls[2] == 2 and rs[2] == 2 else 1
                        point_dict[1] = [(ls[0] + rs[0])//2, (ls[1] + rs[1])//2, indicator]
                else:
                    point_dict[1] = []
                
                limb_dict = {}
                for limb in limb_set:
                    j1, j2 = limb
                    if len(point_dict[j1]) != 0 and len(point_dict[j2]) != 0:
                        limb_dict[limb] = [point_dict[j1], point_dict[j2]]
                    else:
                        limb_dict[limb] = []
                
                ann_list.append((point_dict, limb_dict))

            kp_maps = {}
            paf_maps = {}
            paf_counts = {}
            for ann in ann_list:
                curr_point_dict, curr_limb_dict = ann
                for i in range(18):
                    curr_kp = curr_point_dict[i]
                    if i not in kp_maps:
                        kp_maps[i] = np.zeros(self.end_size)
                    if len(curr_kp) != 0:
                        kp_maps[i] = put_gaussian(curr_kp, kp_maps[i], self.sigma, self.stride)
                
                for limb in limb_set:
                    points = curr_limb_dict[limb]
                    if limb not in paf_maps:
                            paf_maps[limb] = np.zeros((self.end_size[0], self.end_size[1], 2))
                            paf_counts[limb] = np.zeros((self.end_size[0], self.end_size[1]))
                    if len(points) != 0:
                        point1, point2 = tuple(curr_limb_dict[limb])
                        if len(point1) != 0 and len(point2) != 0:
                            updated_pafs, new_counts = put_paf(point1, point2, paf_maps[limb], self.theta, self.stride)
                            paf_maps[limb] = updated_pafs
                            paf_counts[limb] += new_counts
            paf_arr = []
            limbs_sorted = sorted(paf_maps.keys(), key=lambda x: x[0])
            for limb in limbs_sorted:
                curr_map = paf_maps[limb]/paf_counts[limb] if len(paf_counts[limb][paf_counts[limb] != 0]) != 0 else paf_maps[limb]
                curr_map = curr_map.transpose(2,0,1)
                cm1 = aug_det.augment_image(curr_map[0])
                cm2 = aug_det.augment_image(curr_map[1])

                paf_arr.append(np.stack([cm1, cm2]))

            kp_arr_np = np.stack([aug_det.augment_image(x) for _, x in sorted(kp_maps.items(), key=lambda x: x[0])], axis=0)
            paf_arr_np = np.concatenate(paf_arr)
        else:
            kp_arr_np = np.zeros((18, 46, 46))
            paf_arr_np = np.zeros((34, 46, 46))

        curr_img = torch.from_numpy(curr_img.transpose(2,0,1))

        return curr_img.float(), torch.FloatTensor(kp_arr_np), torch.FloatTensor(paf_arr_np)

# base_path = '../data/'
# cocoset = CocoPoseDataset(os.path.join(base_path, 'annotations2017', 'person_keypoints_train2017.json'), os.path.join(base_path, 'train2017'))
# # rand_ind = np.random.randint(len(cocoset))
# # print(rand_ind)

# img, kp_gt, paf_gt = cocoset[97297]
# img, kp_gt, paf_gt = cocoset[97297]

# print(img.size(), kp_gt.size(), paf_gt.size())

# img = F.interpolate(img.unsqueeze(0), size=(46,46), mode='bilinear')

# print(torch.max(kp_gt, 0)[0].size())
# print(torch.max(kp_gt, 0)[0].type())

# utils.save_image(torch.max(kp_gt, 0)[0], 'kp_gt.png')
# utils.save_image(torch.max(torch.abs(paf_gt), 0)[0], 'paf_gt.png')
# utils.save_image(img[0], 'img.png')
