import torch
import torch.nn as nn
from general import bbox_iou


class LocalizeLoss(nn.Module):
    def __init__(self):
        super(LocalizeLoss, self).__init__()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        [:1] : class
        [1:] : box  x y w h
        """
        l_cls = self.criterion_ce(predictions[:, :5], targets[:, 0].long())
        l_bbox = torch.zeros(1).cuda()
        for i, pi in enumerate(predictions):
            l_bbox += torch.sum(torch.square(pi[5:]-targets[i, 1:]))
        total_loss = l_cls + l_bbox
        return total_loss, (l_cls.item(), l_bbox.item())


class LocalizeLoss_Anchor(nn.Module):
    def __init__(self):
        super(LocalizeLoss_Anchor, self).__init__()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        [:1] : class
        [1:] : box  x y w h
        """
        l_cls = sum([self.criterion_ce(predictions[:, i*10:i*10+5], targets[:, 0].long()) for i in range(3)])
        l_bbox = torch.zeros(1).cuda()
        l_conf = torch.zeros(1).cuda()
        for i, pi in enumerate(predictions):
            l_bbox += sum([torch.sum(torch.square(pi[6+j*10:10+j*10]-targets[i, 1:])) for j in range(3)])
            l_conf += sum([torch.square(bbox_iou(pi[6+j*10:10+j*10], targets[i, 1:], x1y1x2y2=False, CIoU=True).detach() - pi[5+j*10]) for j in range(3)])
        total_loss = l_cls + l_bbox + l_conf
        return total_loss, (l_cls.item(), l_bbox.item(), l_conf.item())
