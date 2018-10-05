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

import os

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = rtpose_model(freeze_vgg=False)
    model = model.to(device)

    # print(model)

    if os.path.exists('rtpose.pt'):
        model.load_state_dict(torch.load('rtpose.pt'))

    base_path = '/home/shared/workspace/coco_keypoints'
    cocoset = CocoPoseDataset(os.path.join(base_path, 'annotations'), os.path.join(base_path, 'images'))
    cocoloader = DataLoader(cocoset, batch_size=16, shuffle=True, num_workers=4)

    epochs = 20

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    train_params = filter(lambda x: x.requires_grad, model.parameters())
    opt = optim.SGD(train_params, lr=0.6)

    for e in range(epochs):
        for i, data in enumerate(cocoloader):
            img, kp_gt, paf_gt = data

            img = img.to(device)
            kp_gt = kp_gt.to(device)
            paf_gt = paf_gt.to(device)

            opt.zero_grad()
            last_layer, intermediate_signals = model(img)
            
            curr_loss = 0
            for (signal_kp, signal_paf) in intermediate_signals:
                curr_loss += criterion(signal_kp, kp_gt)
                curr_loss += criterion(signal_paf, paf_gt)
            curr_loss.backward()

            opt.step()

            if i % 100 == 0:
                print('Epoch [%d/%d], Batch [%d/%d], Total Loss %f' % (e, epochs, i, len(cocoloader), curr_loss.item()))

                write_tensor0 = torch.max(last_layer[0][0], 0)[0].unsqueeze(0)
                write_tensor1 = torch.max(kp_gt[0], 0)[0].unsqueeze(0)

                write_tensor2 = torch.max(last_layer[1][0], 0)[0].unsqueeze(0)
                write_tensor3 = torch.max(paf_gt[0], 0)[0].unsqueeze(0)

                img = F.interpolate(img, size=(46,46), mode='bilinear')
                torchvision.utils.save_image(write_tensor0, 'kp_pred.png', nrow=1)
                torchvision.utils.save_image(write_tensor1, 'kp_gt.png', nrow=1)

                torchvision.utils.save_image(write_tensor2, 'paf_pred.png', nrow=1)
                torchvision.utils.save_image(write_tensor3, 'paf_gt.png', nrow=1)
                torchvision.utils.save_image(img[0], 'img.png')
        
        torch.save(model.state_dict(), 'rtpose.pt')

if __name__ == '__main__':
    main()
