from __future__ import print_function

import os
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
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')  # noqa
parser.add_argument('--model', default='./examples/ssd/model/params.pth', type=str, help='initialized model path')  # noqa
parser.add_argument('--checkpoint', default='./examples/ssd/checkpoint/ckpt.pth', type=str, help='checkpoint path')  # noqa
# TODO: Turn off DataParallel so GPU selection works
parser.add_argument('--gpu', default='1', type=int, help='GPU ID (nvidia-smi)')  # noqa
args = parser.parse_args()

NUM_CLASSES = 2
IMG_DIR = '/data/voids/20180215_185312'
LIST_FILE = '/data/voids/20180215_185312.txt'
IMG_DIR_TEST = IMG_DIR
LIST_FILE_TEST = LIST_FILE
BS = 16  # batch size
NUM_EPOCHS = 200
DEBUG = False
SHUFFLE = not DEBUG
NUM_WORKERS = 8 if not DEBUG else 0

# Model
print('==> Building model..')
net = FPNSSD512_2()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 512


def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4,
                                  fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size, img_size),
                        random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


trainset = ListDataset(root=IMG_DIR,
                       list_file=LIST_FILE,
                       transform=transform_train)


def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


testset = ListDataset(root=IMG_DIR_TEST,
                      list_file=LIST_FILE_TEST,
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                          shuffle=SHUFFLE, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False,
                                         num_workers=NUM_WORKERS)

net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

criterion = SSDLoss(num_classes=NUM_CLASSES)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=1e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
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
                 len(trainloader)))


# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1,
                 len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
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
