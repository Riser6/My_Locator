from torchvision.models import resnet50, resnet18
import torch
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, pretrained = True, use_anchor = False):
        super(Detector, self).__init__()
        self.no = 9
        self.use_anchor = use_anchor
        #self.anchor = torch.tensor([[0.46875, 0.359375 ],[0.8046875, 0.703125 ],[0.5078125, 0.8046875]]) #x, y, w, h
        #self.anchor_wh =  torch.tensor([0.64335417, 0.58307292])
        self.backbone = resnet18(pretrained=pretrained)
        fc_inputs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_inputs, self.no)

    def forward(self, x):
        out = self.backbone(x)
        if self.use_anchor:
            self.anchor = self.anchor.cuda()
            #self.anchor_wh = self.anchor_wh.cuda()
            out[:, 5:8] = torch.sigmoid(out[:, 5:8])
            out[:, 8:10] = 2*torch.sigmoid(out[:, 8:10]) * self.anchor[0,:]
            out[:, 15:18] = torch.sigmoid(out[:, 15:18])
            out[:, 18:20] = 2*torch.sigmoid(out[:, 18:20]) * self.anchor[1,:]
            out[:, 25:28] = torch.sigmoid(out[:, 25:28])
            out[:, 28:30] = 2*torch.sigmoid(out[:, 28:30]) * self.anchor[2,:]
            #out[:, 7:9] = 2*torch.sigmoid(out[:, 7:9]) * self.anchor_wh
        else:
            out[:, 5:] = torch.sigmoid(out[:, 5:])
        return out
