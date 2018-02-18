import os
import os.path as osp
import random
import argparse
from time import time

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

from torchcv.models.fpnssd import FPNSSD512_2
from torchcv.models.ssd import SSDBoxCoder
from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import (resize, random_flip, random_paste, random_crop,
                                random_distort)


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')  # noqa
parser.add_argument('--checkpoint', default='checkpoints/ckpt.pth', type=str, help='Where to save and load the checkpoint from')  # noqa
parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
parser.add_argument('--test-code', action='store_true', help='e.g. Reduce sample size')  # noqa
args = parser.parse_args()

TEST_CODE = True  # Only one epoch, only one batch, etc.
TRN_VIDEO_ID = '20180215_185312'
VAL_VIDEO_ID = '20180215_190227'
DATASET_DIR = '../../data/voids'

BATCH_SIZE = 16
NUM_EPOCHS = 200 if not TEST_CODE else 1
IMG_SIZE = 512

DEBUG = False  # Turn off shuffling and multiprocessing
NUM_WORKERS = 8 if not DEBUG else 0

img_dir = osp.join(DATASET_DIR, TRN_VIDEO_ID)
list_file = osp.join(DATASET_DIR, TRN_VIDEO_ID + '.txt')
img_dir_test = osp.join(DATASET_DIR, VAL_VIDEO_ID)
list_file_test = osp.join(DATASET_DIR, VAL_VIDEO_ID + '.txt')
shuffle = not DEBUG
num_classes = 2

# Model
print('==> Building model..')
net = FPNSSD512_2(weights_path='checkpoints/fpnssd512_20_trained.pth')
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
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
    criterion = SSDLoss(num_classes=num_classes)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
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
            if TEST_CODE:
                break

    def test(epoch):
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
            if TEST_CODE:
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
            if not os.path.isdir(os.path.dirname(args.checkpoint)):
                os.mkdir(os.path.dirname(args.checkpoint))
            torch.save(state, args.checkpoint)
            best_loss = test_loss

    start = time()
    for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):
        train(epoch)
        test(epoch)
    print("Minutes elapsed:", (time() - start)/60)
