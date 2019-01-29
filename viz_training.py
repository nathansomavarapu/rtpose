import cv2
import numpy as np
import torch
from visdom import Visdom


class VisdomTrainer():
    def __init__(self, port, hostname):
        self.viz = Visdom(port=port, server=hostname)
        self.win_kp_loss = None
        self.win_paf_loss = None

        self.win_kp_pred_img = None
        self.win_kp_gt_img = None
        self.win_paf_pred_img = None
        self.win_paf_gt_img = None
        self.win_orig = None
        self.viz_counter = 0

        self.init = True


    def update_viz(self, kp_loss_val, paf_loss_val, img, kp_pred_img, kp_gt_img, paf_pred_img, paf_gt_img):

        x_axis = np.array([self.viz_counter])
        kp_data = np.array([kp_loss_val])
        paf_data = np.array([paf_loss_val])


        if self.init :
            self.win_kp_loss = self.viz.line(X=x_axis, Y=kp_data, opts={'linecolor': np.array([[0, 0, 255],]), 'title': 'Keypoints Loss'})
            self.win_paf_loss = self.viz.line(X=x_axis, Y=paf_data, opts={'linecolor': np.array([[255, 165, 0],]), 'title': 'PAF Loss'})
            self.win_kp_pred_img = self.viz.image(kp_pred_img, opts={'title':'Keypoints Predicted Image'})
            self.win_kp_gt_img = self.viz.image(kp_gt_img, opts={'title':'Keypoints GT Image'})
            self.win_paf_pred_img = self.viz.image(paf_pred_img, opts={'title':'PAF Predicted Image'})
            self.win_paf_gt_img = self.viz.image(paf_gt_img, opts={'title':'PAF GT Image'})
            self.win_orig = self.viz.image(img, opts={'title':'Original Image'})

        self.viz.line(X=x_axis, Y=kp_data, win=self.win_kp_loss, update='append')
        self.viz.line(X=x_axis, Y=paf_data, win=self.win_paf_loss, update='append')

        self.viz.image(kp_pred_img, win=self.win_kp_pred_img, opts={'title':'Keypoints Predicted Image'})
        self.viz.image(kp_gt_img, win=self.win_kp_gt_img, opts={'title':'Keypoints GT Image'})
        self.viz.image(paf_pred_img, win=self.win_paf_pred_img, opts={'title':'PAF Predicted Image'})
        self.viz.image(paf_gt_img, win=self.win_paf_gt_img, opts={'title':'PAF GT Image'})
        self.viz.image(img, win=self.win_orig, opts={'title':'Original Image'})

        self.viz_counter += 1
