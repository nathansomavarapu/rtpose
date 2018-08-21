import torchvision.models as models
import torch.nn as nn

class rtmpe(nn.Module):
    
    def __init__(self):
        vgg19 = models.vgg19(pretrained=True)
        self.head = nn.Sequential(*list(vgg19.children())[0][:21])
