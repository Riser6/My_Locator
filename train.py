import argparse
import os
import sys
import torch
import time
import logging

from tiny_vid import Tiny_vid_Dataset
from torch.utils.data import DataLoader
from test import val, val_anchor
from Net import Detector
from loss import LocalizeLoss, LocalizeLoss_Anchor
from tensorboardX import SummaryWriter

import yaml


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
DATAROOT = "tiny_vid/"
HYP_PATH = "hyp.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description='Train Localizer')
    parser.add_argument('--pret',
                        help='load pretrained model',
                        type=str,
                        default=False)
    parser.add_argument('--agum',
                        help='data agumetation',
                        type=str,
                        default=False)
    parser.add_argument('--anchor',
                        help='use anchor',
                        type=str,
                        default=False)                                 
    parser.add_argument('--exp',
                        help='exp name',
                        type=str,
                        default='resnet50_nopre_noagu_noanchor')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='./Datasets')
    parser.add_argument('--batchsize',
                        help='training batchsize',
                        type=int,
                        default=32)
    parser.add_argument('--device',
                        help="training device: 'cpu' or '0' or '0,1,2,3'",
                        type=str,
                        default='0')
    parser.add_argument('--epoch',
                        help="training epoch",
                        type=int,
                        default='300')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(HYP_PATH, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    # bulid up model
    print("begin to bulid up model...")
    torch.cuda.set_device(int(args.device))
    model = Detector(pretrained = args.pret, use_anchor = args.anchor).cuda()
    exp_path = "exp/"+args.exp
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    #logger and tensorboard
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    logging.basicConfig(filename="exp/"+args.exp+"/"+time_str+".log",
                            format='%(asctime)-15s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    writer = SummaryWriter(log_dir ="exp/"+args.exp+"/"+time_str)

    # define loss function (criterion) and optimizer
    criterion = LocalizeLoss()
    #criterion = LocalizeLoss_Anchor()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=hyp["gamma"])

    # Data loading
    print("begin to load data")
    train_dataset = Tiny_vid_Dataset(root=DATAROOT, hyp=hyp, _range=slice(150), mode="train", agument=args.agum)
    test_dataset = Tiny_vid_Dataset(root=DATAROOT, hyp=hyp, _range=slice(150, 180), mode="test")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batchsize)

    print("begin training...")
    best_acc, best_cla_acc, best_bbox_acc, best_epoch= 0, 0, 0, 0
    for epoch in range(args.epoch):
        model.train()
        t0 = time.time()
        train_sum_losses = 0.
        for batch_index, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            total_loss, sub_losses = criterion(outputs, targets)
            train_sum_losses += total_loss
            total_loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Loss {toal_loss:.5f} ({cls_loss:.5f}) ({box_loss:.5f})'.format(
                    epoch, batch_index, len(train_loader), toal_loss=total_loss.item(), cls_loss=sub_losses[0],
                    box_loss=sub_losses[1])
                logger.info(msg)
        train_sum_losses /= len(train_loader)

        writer.add_scalar('train_loss', train_sum_losses.item(), epoch)
        acc, cla_acc, bbox_acc, val_sum_losses= val(test_loader, model)
        #acc, cla_acc, bbox_acc, val_sum_losses= val_anchor(test_loader, model)
        writer.add_scalar('val_loss', val_sum_losses.item(), epoch)
        writer.add_scalar('class_acc', cla_acc, epoch)
        writer.add_scalar('bbox_acc', bbox_acc, epoch)
        print("Time: {}".format(time.time() - t0))
        writer.add_scalar('acc_acc', acc, epoch)
        msg = "Epoch: [{0}]\t train_loss:{1}\t val_loss:{2}\t class_acc:{3}\t bbox_acc:{4}\t acc:{5}".format(
            epoch, train_sum_losses.item(), val_sum_losses.item(), cla_acc, bbox_acc, acc
        )
        logger.info(msg)

        if acc > best_acc:
            best_acc = acc
            best_cla_acc = cla_acc
            best_bbox_acc = bbox_acc
            best_epoch = epoch
            torch.save(model.state_dict(), exp_path+'/best.pth')
        torch.save(model.state_dict(), exp_path+'/last.pth')
        scheduler.step()
    msg = "Best Epoch: [{0}]\t best_acc:{1}\t best_cla_acc:{2}\t best_bbox_acc:{3}".format(
            best_epoch, best_acc, best_cla_acc, best_bbox_acc
        )
    logger.info(msg)
    print("finish!!!")


if __name__ == "__main__":
    main()
