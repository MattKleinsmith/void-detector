import os
import os.path as osp
import random
import argparse
from time import time

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

from torchcv.models.ssd import SSDBoxCoder
from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import (resize, random_flip, random_paste, random_crop,
                                random_distort)

from model import FPNSSD512_2
#from loss import SSDLoss
from utils import set_seed, get_log_prefix


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
parser.add_argument('--test-code', action='store_true', help='Only one epoch, only one batch, etc.')  # noqa
args = parser.parse_args()

TRN_VIDEO_ID = "20180215_185312"
#VAL_VIDEO_ID = "20180215_190227"
VAL_VIDEO_ID = TRN_VIDEO_ID
VOIDS_ONLY = True
RUN_NAME = 'save-based-on-trn'

BATCH_SIZE = 16 if not args.test_code else 2
NUM_EPOCHS = 200 if not args.test_code else 1
IMG_SIZE = 512

DEBUG = False  # Turn off shuffling and multiprocessing
NUM_WORKERS = 8 if not DEBUG else 0
SEED = 123

IMAGE_DIR = "../../data/voids"
LABEL_DIR = "../void-detector/labels"
CKPT_DIR = "checkpoints"

set_seed(SEED)
img_dir = osp.join(IMAGE_DIR, TRN_VIDEO_ID)
voids = "_voids" if VOIDS_ONLY else ''
list_file = osp.join(LABEL_DIR, TRN_VIDEO_ID + voids + '.txt')
print("Training on:", list_file)
img_dir_test = osp.join(IMAGE_DIR, VAL_VIDEO_ID)
list_file_test = osp.join(LABEL_DIR, VAL_VIDEO_ID + voids + '.txt')
print("Testing on:", list_file_test)
shuffle = not DEBUG
num_classes = 2
os.makedirs(CKPT_DIR, exist_ok=True)

# Model
print('==> Building model..')

net = FPNSSD512_2(weights_path='checkpoints/fpnssd512_20_trained.pth')
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)


def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4,
                                  fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(IMG_SIZE, IMG_SIZE),
                        random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(IMG_SIZE, IMG_SIZE))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


trn_ds = ListDataset(root=img_dir,
                     list_file=list_file,
                     transform=transform_train)
val_ds = ListDataset(root=img_dir_test, list_file=list_file_test,
                     transform=transform_test)
trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=BATCH_SIZE,
                                     shuffle=shuffle, num_workers=NUM_WORKERS)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=NUM_WORKERS)
with torch.cuda.device(args.gpu):
    net.cuda()
    cudnn.benchmark = True
    criterion = SSDLoss(num_classes=2)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=1e-4)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trn_dl):
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1,
                     len(trn_dl)))
            if args.test_code:
                break

    def test(epoch, log_prefix):
        print('\nTest')
        net.eval()
        test_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(val_dl):
            inputs = Variable(inputs.cuda(), volatile=True)
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.data[0]
            print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1,
                     len(val_dl)))
            if args.test_code:
                test_loss = 0
                break

        # Save checkpoint
        global best_loss
        test_loss /= len(val_dl)
        if test_loss < best_loss:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'loss': test_loss,
                'epoch': epoch,
            }
            suffix = ('epoch-%03d' % epoch) + '.pth'
            ckpt_path = osp.join(CKPT_DIR, log_prefix + suffix)
            torch.save(state, ckpt_path)
            print(ckpt_path)
            best_loss = test_loss

    suffix = '_' + RUN_NAME + '_' if RUN_NAME else ''
    log_prefix = get_log_prefix() + suffix
    start = time()
    for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):
        train(epoch)
        test(epoch, log_prefix)
    print("Minutes elapsed:", (time() - start)/60)
