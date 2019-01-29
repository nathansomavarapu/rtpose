import cv2
import numpy as np
import torch
from visdom import Visdom


class VisdomTrainer():
    def __init__(self, port, hostname):
        self.viz = Visdom(port=port, server=hostname)
        self.win_kp_loss = None
        self.win_paf_loss = None

        self.win_match = None
        self.win_ann = None
        self.win_pred = None
        self.text_window_tr = None
        self.text_window_pred = None
        self.viz_counter = 0

        self.init = True


    def update_viz(self, kp_loss_val, paf_loss_val, img, kp_pred_img, kp_gt_img, paf_pred_img, paf_pred_img):

        x_axis = np.array([self.viz_counter])
        kp_data = np.array([kp_loss_val])
        paf_data = np.array([paf_loss_val])


        if self.init :
            self.win_kp_loss = self.viz.line(X=x_axis, Y=kp_data, opts={'linecolor': np.array([[0, 0, 255],]), 'title': 'Keypoints Loss'})
            self.win_loc = self.viz.line(X=x_axis, Y=paf_data, opts={'linecolor': np.array([[255, 165, 0],]), 'title': 'PAF Loss'})
            self.win_match = self.viz.image(match_img, opts={'title':'Match Image'})
            self.win_pred = self.viz.image(pred_img, opts={'title':'Predictions Image'})
            self.text_window_tr = self.viz.text(true_cl_str)
            self.text_window_pred = self.viz.text(pred_cl_str)

        self.viz.line(X=x_axis, Y=cl_data, win=self.win_loss, update='append')

        self.viz.image(match_img, win=self.win_match, opts={'title':'Match Image'})
        self.viz.image(pred_img, win=self.win_pred, opts={'title':'Predictions Image'})

        self.viz.text(true_cl_str, win=self.text_window_tr)
        self.viz.text(pred_cl_str, win=self.text_window_pred)

        self.viz_counter += 1
