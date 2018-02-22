import os
import os.path as osp
import random
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm, trange

from torchcv.models.ssd import SSDBoxCoder
#from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import (resize, random_flip, random_paste, random_crop,
                                random_distort)

from torchcv.models.void_models import FPNSSD512_2
from torchcv.loss.void_losses import SSDLoss
from utils import set_seed, get_log_prefix


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
parser.add_argument('--test-code', action='store_true', help='Only one epoch, only one batch, etc.')  # noqa
args = parser.parse_args()

TRN_VIDEO_ID = "20180215_185312"
#VAL_VIDEO_ID = "20180215_190227"
VAL_VIDEO_ID = TRN_VIDEO_ID
VOIDS_ONLY = False
RUN_NAME = 'save-based-on-trn_voidless-included'

BATCH_SIZE = 16 if not args.test_code else 2
NUM_EPOCHS = 300 if not args.test_code else 2
IMG_SIZE = 512

DEBUG = False  # Turn off shuffling and multiprocessing
NUM_WORKERS = 8 if not DEBUG else 0
SEED = 123

IMAGE_DIR = "../../data/voids"
LABEL_DIR = "../void-detector/labels"
CKPT_DIR = "checkpoints"

print("Run name:", RUN_NAME)
set_seed(SEED)
img_dir = osp.join(IMAGE_DIR, TRN_VIDEO_ID)
voids = "_voids" if VOIDS_ONLY else ''
list_file = osp.join(LABEL_DIR, TRN_VIDEO_ID + voids + '.txt')
print("Training on:", list_file)
img_dir_test = osp.join(IMAGE_DIR, VAL_VIDEO_ID)
list_file_test = osp.join(LABEL_DIR, VAL_VIDEO_ID + voids + '.txt')
print("Testing on:", list_file_test)
shuffle = not DEBUG
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

if args.test_code:
    trn_ds.num_imgs = 10
    val_ds.num_imgs = 10

with torch.cuda.device(args.gpu):
    net.cuda()
    cudnn.benchmark = True  # WARNING: Don't use if using images w/ diff shapes  # TODO: Check for this condition automatically
    criterion = SSDLoss()
    lr = 1e-3
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=1e-4)

    def train(epoch):
        net.train()
        trn_loss = 0
        tqdm_trn_dl = tqdm(trn_dl, desc="Train", ncols=0)
        for batch_idx, batch in enumerate(tqdm_trn_dl):
            inputs, loc_targets, cls_targets = batch
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trn_loss += loss.data[0]
            avg_loss = trn_loss/(batch_idx+1)
            # tqdm_trn_dl.set_postfix(avg_loss="{:.2f}".format(avg_loss))

        return avg_loss

    def validate(epoch, log_prefix, run_name, tqdm_epochs, trn_avg_loss):
        net.eval()
        val_loss = 0
        tqdm_val_dl = tqdm(val_dl, desc="Validate", ncols=0)
        for batch_idx, batch in enumerate(tqdm_val_dl):
            inputs, loc_targets, cls_targets = batch
            inputs = Variable(inputs.cuda(), volatile=True)
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)

            val_loss += loss.data[0]
            avg_loss = val_loss/(batch_idx+1)
            # tqdm_val_dl.set_postfix(avg_loss="{:.2f}".format(avg_loss))

        if args.test_code:
            val_loss = 0

        # Save checkpoint
        global best_loss
        val_loss /= len(val_dl)
        if val_loss < best_loss:
            state = {
                'net': net.state_dict(),  # TODO: Switch to 'state_dict'
                'loss': val_loss,
                'epoch': epoch,
            }
            values = [epoch, trn_avg_loss, avg_loss, lr]
            layout = "_epochs-{:03d}_trn_loss-{:.6f}_val_loss-{:.6f}"
            layout += "_lr-{:.2E}"
            suffix = layout.format(*values) + run_name + '.pth'
            ckpt_path = osp.join(CKPT_DIR, log_prefix + suffix)
            torch.save(state, ckpt_path)
            tqdm_epochs.write(ckpt_path)
            best_loss = val_loss

    log_prefix = get_log_prefix()
    tqdm_epochs = trange(start_epoch, start_epoch+NUM_EPOCHS, desc="Epoch",
                         ncols=0)
    for epoch in tqdm_epochs:
        trn_avg_loss = train(epoch)
        validate(epoch, log_prefix, RUN_NAME, tqdm_epochs, trn_avg_loss)
