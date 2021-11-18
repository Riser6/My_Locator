import os
import torch
import cv2

from general import bbox_iou
import numpy as np
from Net import Detector
from tiny_vid import Tiny_vid_Dataset
from torch.utils.data import DataLoader
from loss import LocalizeLoss, LocalizeLoss_Anchor

classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
root = "tiny_vid/"


def val(test_loader, model):
    model.eval()
    ious = []
    cla_preds = []
    cla_gts = []
    criterion = LocalizeLoss()
    with torch.no_grad():
        val_sum_losses = 0.
        for batch_index, (_, imgs, targets) in enumerate(test_loader):
            imgs, targets = imgs.cuda(), targets.cuda()
            predictions = model(imgs)
            total_loss, sub_losses = criterion(predictions, targets)
            
            val_sum_losses += total_loss
            cla_pred = torch.argmax(predictions[:, :5], dim=1)
            cla_gt = targets[:, 0]
            cla_preds.append(cla_pred.cpu().numpy())
            cla_gts.append(cla_gt.cpu().numpy())

            for i, pi in enumerate(predictions):
                iou = bbox_iou(pi[5:], targets[i, 1:], x1y1x2y2=False)  # iou(prediction, target)
                ious.append(iou.cpu().item())

    cla_preds = np.concatenate(cla_preds, axis=0)
    cla_gts = np.concatenate(cla_gts, axis=0)
    cla_acc = np.sum(cla_gts == cla_preds) / cla_gts.shape[0]
    ious = np.array(ious)
    bbox_acc = np.sum(ious > 0.5) * 1.0 / ious.shape[0]
    acc = np.sum((ious > 0.5) & (cla_gts == cla_preds)) * 1.0 / ious.shape[0]
    model.train()
    val_sum_losses /= len(test_loader)
    return acc, cla_acc, bbox_acc, val_sum_losses


def val_anchor(test_loader, model):
    model.eval()
    ious = []
    cla_preds = []
    cla_gts = []
    criterion = LocalizeLoss_Anchor()
    with torch.no_grad():
        val_sum_losses = 0.
        for batch_index, (_, imgs, targets) in enumerate(test_loader):
            imgs, targets = imgs.cuda(), targets.cuda()
            predictions = model(imgs)
            total_loss, sub_losses = criterion(predictions, targets)
            val_sum_losses += total_loss
            p1,p2,p3 = predictions.chunk(3,1)
            predictions = torch.stack([p1,p2,p3], dim=1)
            _, indices = torch.max(predictions[:,:,5],1)
            predictions = torch.stack([predictions[i,idc,:] for i, idc in enumerate(indices)] ,dim=0)
            cla_pred = torch.argmax(predictions[:, :5], dim=1)
            cla_gt = targets[:, 0]
            cla_preds.append(cla_pred.cpu().numpy())
            cla_gts.append(cla_gt.cpu().numpy())

            for i, pi in enumerate(predictions):
                iou = bbox_iou(pi[6:], targets[i, 1:], x1y1x2y2=False)  # iou(prediction, target)
                ious.append(iou.cpu().item())

    cla_preds = np.concatenate(cla_preds, axis=0)
    cla_gts = np.concatenate(cla_gts, axis=0)
    cla_acc = np.sum(cla_gts == cla_preds) / cla_gts.shape[0]
    ious = np.array(ious)
    bbox_acc = np.sum(ious > 0.5) * 1.0 / ious.shape[0]
    acc = np.sum((ious > 0.5) & (cla_gts == cla_preds)) * 1.0 / ious.shape[0]
    model.train()
    val_sum_losses /= len(test_loader)
    return acc, cla_acc, bbox_acc, val_sum_losses


def inference(test_loader, model):
    model.eval()
    model.cuda()
    if not os.path.exists("output"):
        os.mkdir("output")
    cls_r = 0
    cls_total = 0
    print(len(test_loader))
    with torch.no_grad():
        for index, image, labels in test_loader:
            cls_total += 1
            index = index[0].zfill(6)
            image = image.cuda()
            pred = model(image).squeeze().cpu()  # 9
            cls = torch.argmax(pred[:5]).long()
            cls_gt = labels.squeeze()[0]
            iou = bbox_iou(pred[5:], labels[0, 1:], x1y1x2y2=False)

            print(cls)
            print(cls_gt)
            if cls == cls_gt:
                cls_r += 1

            cls_str = classes[cls]
            gt_cls_str = classes[int(cls_gt)]
            x, y, w, h = pred[5:]
            x *= 128
            y *= 128
            w *= 128
            h *= 128
            image = cv2.imread(os.path.join(root, gt_cls_str, index + ".JPEG"))
            if iou > 0.5:
                plot = cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)
            else:
                plot = cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if cls == cls_gt:
                plot = cv2.putText(plot, cls_str, (int(x - w/2), int(y)), font, 0.5, (255, 255, 255), 2)
            else:
                plot = cv2.putText(plot, cls_str, (int(x - w/2), int(y)), font, 0.5, (0, 0, 255), 2)

            if not os.path.exists(os.path.join("output", gt_cls_str)):
                os.mkdir(os.path.join("output", gt_cls_str))
            cv2.imwrite(os.path.join("output", gt_cls_str, index + ".JPEG"), plot)
            # print(index)
        print(cls_r * 1.0 / cls_total)


if __name__ == "__main__":
    model = Detector()
    model.load_state_dict(torch.load("exp/resnet18_pre_agu_noanchor/best.pth"))
    test_dataset = Tiny_vid_Dataset(root=root, _range=slice(150, 180), mode="test")
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    inference(test_loader, model)
