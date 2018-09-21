from cocoloader import CocoPoseDataset
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rtpose import rtpose_model
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def gen_loss(intermediate_signals, kp_gt, paf_gt, device, criterion):
    total_loss = None
    each_loss = []

    for i in range(len(intermediate_signals)//2):
        kp_i = intermediate_signals[2*i]
        paf_i = intermediate_signals[2*i+1]

        kp_loss_i = criterion(kp_i, kp_gt)
        paf_loss_i = criterion(paf_i, paf_gt)

        if total_loss is None:
            total_loss = kp_loss_i + paf_loss_i
        else:
            total_loss += kp_loss_i + paf_loss_i
        each_loss.append((kp_loss_i.item(), paf_loss_i.item()))
    
    return total_loss, each_loss


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = rtpose_model(freeze_vgg=False)
    # model = model.to(device)

    base_path = '/home/shared/workspace/coco_keypoints'
    cocoset = CocoPoseDataset(os.path.join(base_path, 'annotations'), os.path.join(base_path, 'images'))
    cocoloader = DataLoader(cocoset, batch_size=4, shuffle=True, num_workers=4)


    epochs = 5

    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    train_params = filter(lambda x: x.requires_grad, model.parameters())
    opt = optim.SGD(train_params, lr=1)

    for e in range(epochs):
        for i, data in enumerate(cocoloader):
            img, kp_gt, paf_gt = data

            # TODO: Need to set list of stages to self in the model, so that *.to(device) works

            # img = img.to(device)
            # kp_gt = kp_gt.to(device)
            # paf_gt = paf_gt.to(device)

            last_layer, intermediate_signals = model(img)

            opt.zero_grad()
            curr_loss, all_losses = gen_loss(intermediate_signals, kp_gt, paf_gt, device, criterion)
            curr_loss.backward()

            opt.step()

            if i % 100 == 0:
                print('Epoch [%d/%d], Batch [%d/%d], Total Loss %f' % (e, epochs, i, len(cocoset), curr_loss.item()))

if __name__ == '__main__':
    main()
