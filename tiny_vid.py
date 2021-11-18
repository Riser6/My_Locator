import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import random
from PIL import Image
import cv2

from augmentations import Albumentations, augment_hsv, random_perspective

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

classes = ['car', 'bird', 'turtle', 'dog', 'lizard']


class Tiny_vid_Dataset(Dataset):
    def __init__(self, root, hyp=None, _range=None, mode="train", agument=True):
        self.imgs = []
        self.labels = []
        self.idxs = []
        self.anchors = [0.5, 0.39751562, 0.64335417, 0.58307292]
        self.transform = transform
        self.hyp = hyp
        self.mode = mode
        self.agument = agument
        self.albumentations = Albumentations()
        assert mode in ["train", "test"]
        if mode == "train":
            for i, cls in enumerate(classes):
                folder = os.path.join(root, cls)
                txt = os.path.join(root, cls + "_gt.txt")
                with open(txt) as f:
                    for line in f.readlines()[_range]:
                        idx, x1, y1, x2, y2 = line.strip().split()
                        self.imgs.append(os.path.join(folder, idx.zfill(6)) + ".JPEG")
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                        self.labels.append([i, x1, y1, x2, y2])
        else:
            for i, cls in enumerate(classes):
                folder = os.path.join(root, cls)
                txt = os.path.join(root, cls + "_gt.txt")
                with open(txt) as f:
                    for line in f.readlines()[_range]:
                        idx, x1, y1, x2, y2 = line.strip().split()
                        self.idxs.append(idx)
                        self.imgs.append(os.path.join(folder, idx.zfill(6)) + ".JPEG")
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                        self.labels.append([i, x1, y1, x2, y2])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        cls, x1, y1, x2, y2 = self.labels[index]
        label = np.asarray([cls, x1, y1, x2, y2])
        hyp = self.hyp

        if self.mode == "train":
            if self.agument:
                labels = np.expand_dims(label, axis=0)
                img, labels = random_perspective(img, labels,
                                                degrees=hyp['degrees'],
                                                translate=hyp['translate'],
                                                scale=hyp['scale'],
                                                shear=hyp['shear'],
                                                perspective=hyp['perspective'])
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])
                img, labels = self.albumentations(img, labels)
                nl = len(labels)  # update after albumentations

                # HSV color-space
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

                # Flip up-down
                if random.random() < hyp['flipud']:
                    img = np.flipud(img)
                    if nl:
                        labels[:, 2] = 1 - labels[:, 2]

                # Flip left-right
                if random.random() < hyp['fliplr']:
                    img = np.fliplr(img)
                    if nl:
                        labels[:, 1] = 1 - labels[:, 1]
                label = np.squeeze(labels, 0)


            else:
                my_label = np.random.random(5)
                my_label[1:5] = xyxy2xywh(label[1:5], w=img.shape[1], h=img.shape[0])
                my_label[0] = label[0]
                my_label = np.random.random(5)
                my_label[1:5] = xyxy2xywh(label[1:5], w=img.shape[1], h=img.shape[0])
                my_label[0] = label[0]
                if self.transform is not None:
                    img = self.transform(img)    
                return img, my_label

                #visualization
                """x,y,w,h = label[1:]
                cls = int(label[0])
                cls_str = classes[cls]
                x1, y1, x2, y2 = int((x - w/2)*128), int((y - h/2)*128), int((x + w/2)*128), int((y + h/2)*128)
                img = np.clip(img, 0, 255).astype(np.uint8)
                plot = cv2.rectangle(img, (x1,y1), (x2, y2), (0, 255, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                plot = cv2.putText(plot, cls_str, (x1, int(y1)), font, 0.5, (255, 255, 255), 2)
                cv2.imwrite(str(index)+"gt.JPEG",plot)"""

            img = np.ascontiguousarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        else:
            my_label = np.random.random(5)
            my_label[1:5] = xyxy2xywh(label[1:5], w=img.shape[1], h=img.shape[0])
            my_label[0] = label[0]
            if self.transform is not None:
                img = self.transform(img)
            return self.idxs[index], img, my_label


def xyxy2xywhn(x, w=128, h=128, clip=False, eps=0.0):
    y = np.random.random((1,4))
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def xyxy2xywh(x, w=128, h=128, clip=False, eps=0.0):
    y = np.random.random(4)
    y[0] = ((x[0] + x[2]) / 2) / w  # x center
    y[1] = ((x[1] + x[3]) / 2) / h  # y center
    y[2] = (x[2] - x[0]) / w  # width
    y[3] = (x[3] - x[1]) / h  # height
    return y



