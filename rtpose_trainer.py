from cocoloader import CocoPoseDataset
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rtpose import rtpose_model
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import torch.nn.functional as F

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = rtpose_model(freeze_vgg=False)
    model = model.to(device)

    base_path = '/home/shared/workspace/coco_keypoints'
    cocoset = CocoPoseDataset(os.path.join(base_path, 'annotations'), os.path.join(base_path, 'images'))
    cocoloader = DataLoader(cocoset, batch_size=4, shuffle=True, num_workers=4)

    epochs = 5

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    train_params = filter(lambda x: x.requires_grad, model.parameters())
    opt = optim.SGD(train_params, lr=1)

    test_ind = np.random.randint(len(cocoset))
    print(test_ind)
    img, kp_gt = cocoset[np.random.randint(len(cocoset))]
    torchvision.utils.save_image(img, 'img_orig.png')
    write_tensor1 = torch.max(kp_gt, 0)[0]
    img = F.upsample(img.unsqueeze(0), size=(46,46), mode='bilinear')
    img = img[0]
    img[0] = torch.where(write_tensor1 > 0, img[0] * write_tensor1, torch.zeros(46,46))
    img[1] = torch.where(write_tensor1 > 0, img[1] * write_tensor1, torch.zeros(46,46))
    img[2] = torch.where(write_tensor1 > 0, img[2] * write_tensor1, torch.zeros(46,46))

    torchvision.utils.save_image(write_tensor1, 'sample_gt.png', nrow=3)
    torchvision.utils.save_image(img, 'img.png')

    # for e in range(epochs):
    #     for i, data in enumerate(cocoloader):

    #         print(len(data))
    #         img, kp_gt = data

    #         img = img.to(device)
    #         kp_gt = kp_gt.to(device)

    #         last_layer, intermediate_signals = model(img)

    #         opt.zero_grad()
    #         curr_loss = 0
    #         for signal in intermediate_signals:
    #             curr_loss += criterion(signal, kp_gt)
    #         curr_loss.backward()

    #         opt.step()

    #         if i % 100 == 0:
    #             print('Epoch [%d/%d], Batch [%d/%d], Total Loss %f' % (e, epochs, i, len(cocoset), curr_loss.item()))

    #             write_tensor0 = last_layer[0][0].unsqueeze(0).permute(1,0,2,3)
    #             write_tensor1 = kp_gt[0].unsqueeze(0).permute(1,0,2,3)

    #             torchvision.utils.save_image(write_tensor0, 'sample_pred.png', nrow=3)
    #             torchvision.utils.save_image(write_tensor1, 'sample_gt.png', nrow=3)
    #             torchvision.utils.save_image(img[0], 'img.png')

if __name__ == '__main__':
    main()
