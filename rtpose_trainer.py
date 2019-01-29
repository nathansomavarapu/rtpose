from cocoloader import CocoPoseDataset
import os

import torch
from torch.autograd import Variable
from rtpose import rtpose_model
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

from viz_training import VisdomTrainer

import os

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    enable_viz = True
    port = 8908
    hostname = 'http://localhost'

    model = rtpose_model(freeze_vgg=True, reinit_vgg=False)
    model = model.to(device)

    if os.path.exists('rtpose.pt'):
        model.load_state_dict(torch.load('rtpose.pt'))

    base_path = '../data'
    cocoset = CocoPoseDataset(os.path.join(base_path, 'annotations2017/person_keypoints_train2017.json'), os.path.join(base_path, 'train2017'))
    cocoloader = DataLoader(cocoset, batch_size=32, shuffle=True, num_workers=4)

    epochs = 200

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    train_params = filter(lambda x: x.requires_grad, model.parameters())
    opt = optim.Adam(train_params, lr=0.9)

    viz = None
    if enable_viz:
        viz = VisdomTrainer(port, hostname)


    for e in range(epochs):
        for i, data in enumerate(cocoloader):
            img, kp_gt, paf_gt = data

            img = img.to(device)
            kp_gt = kp_gt.to(device)
            paf_gt = paf_gt.to(device)

            last_layer, intermediate_signals = model(img)
            
            
            kp_loss = 0
            paf_loss = 0
            for (signal_kp, signal_paf) in intermediate_signals:
                kp_loss += criterion(signal_kp, kp_gt)
                paf_loss += criterion(signal_paf, paf_gt)
            
            curr_loss = kp_loss + paf_loss

            opt.zero_grad()
            curr_loss.backward()
            opt.step()
            

            if i % 100 == 0 and viz is not None:

                img = img[0].data.cpu().numpy()

                write_tensor0 = torch.max(last_layer[0][0].data, 0)[0].unsqueeze(0).cpu().numpy()
                write_tensor1 = torch.max(kp_gt[0].data, 0)[0].unsqueeze(0).cpu().numpy()

                write_tensor2 = torch.max(torch.abs(last_layer[1][0].data), 0)[0].unsqueeze(0).cpu().numpy()
                write_tensor3 = torch.max(torch.abs(paf_gt[0].data), 0)[0].unsqueeze(0).cpu().numpy()

                viz.update_viz(kp_loss.item(), paf_loss.item(), img, write_tensor0, write_tensor1, write_tensor2, write_tensor3)

    torch.save(model.state_dict(), 'rtpose.pt')

if __name__ == '__main__':
    main()
