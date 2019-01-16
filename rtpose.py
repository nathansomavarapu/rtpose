import torchvision.models as models
import torch.nn as nn
import torch


class rtpose_model(nn.Module):
    
    def __init__(self, freeze_vgg=False, reinit_vgg=False):
        super(rtpose_model, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        vgg_layers = list(vgg19.features[:19])
        vgg_layers.extend([nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)])
        self.head = nn.Sequential(*vgg_layers)

        self.stages = {}

        self.s1_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 18, 1, padding=0),
        )
        self.stages['s1_1'] = getattr(self, 's1_1')

        self.s1_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 34, 1, padding=0),
        )
        self.stages['s1_2'] = getattr(self, 's1_2')

        for i in range(2,7):
            
            curr_s1_1 = nn.Sequential(
                nn.Conv2d(180, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 18, 1, padding=0),
            )
            setattr(self, 's' + str(i) + '_1', curr_s1_1)
            self.stages['s' + str(i) + '_1'] = getattr(self, 's' + str(i) + '_1')

            curr_s1_2 = nn.Sequential(
                nn.Conv2d(180, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 34, 1, padding=0),
            )
            setattr(self, 's' + str(i) + '_2', curr_s1_2)
            self.stages['s' + str(i) + '_2'] = getattr(self, 's' + str(i) + '_2')
        
        self.init_weights(reinit_vgg)

        if freeze_vgg:
            for lay in self.head.children():
                for param in lay.parameters():
                    param.requires_grad = False


    def forward(self, x):
        x0 = self.head(x)
        inter_signals = []

        prev = x0
        x_new1 = None
        x_new2 = None
        for i in range(1,7):
            x_new1 = getattr(self, 's' + str(i) + '_1')(prev)
            x_new2 = getattr(self, 's' + str(i) + '_2')(prev)

            prev = torch.cat([x_new1, x_new2, x0], 1)
            inter_signals.append((x_new1, x_new2))
        
        return (x_new1, x_new2), inter_signals
    
    def init_weights(self, reinit_vgg):

        if reinit_vgg:
            for layer in self.head:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        
        for stage_k in self.stages.keys():
            curr_stage = self.stages[stage_k]
            for layer in curr_stage:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
