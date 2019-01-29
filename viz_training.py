import cv2
import numpy as np
import torch
from visdom import Visdom


class VisdomTrainer():
    def __init__(self, port, hostname):
        self.viz = Visdom(port=port, server=hostname)
        self.win_loss = None

        self.win_match = None
        self.win_ann = None
        self.win_pred = None
        self.text_window_tr = None
        self.text_window_pred = None
        self.viz_counter = 0


    def update_viz(self, cl_loss_val, loc_loss_val, img, default_boxes, match_idxs, ann_cl, ann_boxes, predicted_classes, predicted_offsets):

        x_axis = np.array([self.viz_counter])
        cl_data = np.array([cl_loss_val])

        match_img = self.get_match_img(img, default_boxes, match_idxs, ann_boxes)
        pred_img = self.get_pred_img(img, default_boxes, predicted_offsets, non_zero_pred_idxs)

        true_cl_str = 'True Classes: ' + str(ann_classes)
        pred_cl_str = 'Predicted Classes: ' + str(pred_classes)

        if self.win_loss is None or self.win_match is None:
            self.win_loss = self.viz.line(X=x_axis, Y=cl_data, opts={'linecolor': np.array([[0, 0, 255],]), 'title': 'Classification Loss'})
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
