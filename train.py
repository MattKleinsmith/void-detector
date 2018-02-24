from glob import glob
import argparse
import os
import os.path as osp
import random
import sqlite3
from time import time

from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torchcv.models.ssd import SSDBoxCoder
from torchcv.datasets import ListDataset
from torchcv.transforms import (resize, random_flip, random_paste, random_crop,
                                random_distort)

from torchcv.models.void_models import FPNSSD512_2
from torchcv.loss.void_losses import SSDLoss
from utils import (set_seed, get_log_prefix, videoid2videoname, git_hash,
                   get_datetime, get_gpu_names)
from utils.sql import get_trial_id, save_stats, connect_and_execute
from evaluate import evaluate


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
parser.add_argument('--test-code', action='store_true', help='Only one epoch, only one batch, etc.')  # noqa
parser.add_argument('--video-id', default=0, type=int, choices=[-1, 0, 1])  # noqa
parser.add_argument('--include-voidless', action='store_true', help='Include voidless images')  # noqa
args = parser.parse_args()

video_name = videoid2videoname(args.video_id)
TRN_VIDEO_ID = video_name
VAL_VIDEO_ID = TRN_VIDEO_ID
RUN_NAME = 'save-based-on-trn'
if args.include_voidless:
    RUN_NAME = "voidless-included_" + RUN_NAME

BATCH_SIZE = 16 if not args.test_code else 2
NUM_EPOCHS = 300 if not args.test_code else 1
IMG_SIZE = 512

DEBUG = False  # Turn off shuffling and multiprocessing
NUM_WORKERS = 8 if not DEBUG else 0
SEED = 123
TRACK_BOX_EVOLUTION = True

IMAGE_DIR = "../../data/voids"
LABEL_DIR = "../void-detector/labels"
CKPT_DIR = "checkpoints"
ARCH = FPNSSD512_2
ARCH_KWARGS = {'weights_path': 'checkpoints/fpnssd512_20_trained.pth'}

print("Run name:", RUN_NAME)
set_seed(SEED)
img_dir = osp.join(IMAGE_DIR, TRN_VIDEO_ID)
voids = "_voids" if not args.include_voidless else ''
trn_labels_fpath = osp.join(LABEL_DIR, TRN_VIDEO_ID + voids + '.txt')
print("Training set:", trn_labels_fpath)
img_dir_test = osp.join(IMAGE_DIR, VAL_VIDEO_ID)
val_labels_fpath = osp.join(LABEL_DIR, VAL_VIDEO_ID + voids + '.txt')
print("Validation set:", val_labels_fpath)
shuffle = not DEBUG
os.makedirs(CKPT_DIR, exist_ok=True)
gpu_name = get_gpu_names()[args.gpu]
trn_name = osp.basename(trn_labels_fpath)
val_name = osp.basename(val_labels_fpath)
arch_name = ARCH.__name__

sqlite_path = "database.sqlite3"
trial_id = get_trial_id(sqlite_path) if not args.test_code else -1
git = git_hash()
print("Trial ID:", trial_id)

with torch.cuda.device(args.gpu):
    # Model
    print('==> Building model..')
    net = ARCH(**ARCH_KWARGS)
    net.cuda()
    cudnn.benchmark = True  # WARNING: Don't use if using images w/ diff shapes  # TODO: Check for this condition automatically
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch
    criterion = SSDLoss()
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)

    # Dataset
    print('==> Preparing dataset..')
    box_coder = SSDBoxCoder(net)

    def trn_transform(img, boxes, labels):
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

    def val_transform(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(IMG_SIZE, IMG_SIZE))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        boxes, labels = box_coder.encode(boxes, labels)
        return img, boxes, labels

    trn_ds = ListDataset(root=img_dir, list_file=trn_labels_fpath,
                         transform=trn_transform, test_code=args.test_code)
    val_ds = ListDataset(root=img_dir_test, list_file=val_labels_fpath,
                         transform=val_transform, test_code=args.test_code)
    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=BATCH_SIZE,
                                         shuffle=shuffle,
                                         num_workers=NUM_WORKERS)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=NUM_WORKERS)

    def calculate_loss(batch_idx, batch, volatile=False):
        inputs, loc_targets, cls_targets = batch
        inputs = Variable(inputs.cuda(), volatile=volatile)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        return loss

    def train(epoch):
        net.train()
        trn_loss = 0
        tqdm_trn_dl = tqdm(trn_dl, desc="Train", ncols=0)
        for batch_idx, batch in enumerate(tqdm_trn_dl):
            loss = calculate_loss(batch_idx, batch, volatile=False)
            trn_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return trn_loss / len(trn_dl)

    def validate(epoch, log_prefix, run_name, tqdm_epochs, trn_loss):
        net.eval()
        val_loss = 0
        tqdm_val_dl = tqdm(val_dl, desc="Validate", ncols=0)
        for batch_idx, batch in enumerate(tqdm_val_dl):
            loss = calculate_loss(batch_idx, batch, volatile=True)
            val_loss += loss.data[0]
        # Save checkpoint
        global best_loss
        val_loss /= len(val_dl)
        if val_loss < best_loss or args.test_code:
            ckpt = {
                'trial_id': trial_id,
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict()}
            log_prefix = "trial-{:04d}_".format(trial_id) + log_prefix
            values = [epoch, trn_loss, val_loss, lr, BATCH_SIZE,
                      IMG_SIZE]
            layout = "_epoch-{:03d}_trn_loss-{:.6f}"
            layout += "_val_loss-{:.6f}_lr-{:.2E}_bs-{:03d}_sz-{}_"
            suffix = layout.format(*values) + run_name + '.pth'
            ckpt_path = osp.join(CKPT_DIR, log_prefix + suffix)
            torch.save(ckpt, ckpt_path)
            ckpt_paths = glob("checkpoints/trial-{:04d}*".format(trial_id))
            for path in list(set(ckpt_paths) - set([ckpt_path])):
                os.remove(path)
            tqdm_epochs.write(ckpt_path)
            best_loss = val_loss
        # Save stats to database
        stats = dict(
            trial_id=trial_id, datetime=get_datetime(), git=git, epoch=epoch,
            trn_loss=trn_loss, val_loss=val_loss,
            num_trn=len(trn_ds), num_val=len(val_ds),
            trn_name=trn_name, val_name=val_name,
            voidless_included=args.include_voidless,
            arch=ARCH, loss_fn=criterion.__class__,
            optimizer=optimizer.__class__,
            lr=lr, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
            momentum=momentum, weight_decay=weight_decay, seed=SEED,
            gpu_name=gpu_name, timestamp=time())
        save_stats(sqlite_path, stats)
        if TRACK_BOX_EVOLUTION:
            cls_id = 0  # voids
            img = Image.open("docs/20180215_190227_002190.jpg")
            x = img.resize((IMG_SIZE, IMG_SIZE))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            x = transform(x)
            x = Variable(x, volatile=True).cuda()
            loc_preds, cls_preds = net(x.unsqueeze(0))
            boxes, labels, scores = box_coder.decode(
                loc_preds.data.squeeze().cpu(),
                F.softmax(cls_preds.squeeze(), dim=1).data.cpu(),
                nms_thresh=1.0, score_thresh=0.22)
            boxes = [box for i, box in enumerate(boxes) if labels[i] == cls_id]
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                stats['x_min'] = x1
                stats['y_min'] = y1
                stats['x_max'] = x2
                stats['y_max'] = y2
                stats['score'] = scores[i]
                stats['timestamp'] = time()
                save_stats(sqlite_path, stats)
        return val_loss, stats

    if TRACK_BOX_EVOLUTION:
        # This makes an awesome image but it takes too long to do it every
        # time.
        stats = dict(trial_id=trial_id, datetime=get_datetime(),
                     git=git, epoch=-1)
        cls_id = 0  # voids
        img = Image.open("docs/20180215_190227_002190.jpg")
        x = img.resize((IMG_SIZE, IMG_SIZE))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        x = transform(x)
        x = Variable(x, volatile=True).cuda()
        loc_preds, cls_preds = net(x.unsqueeze(0))
        boxes, labels, scores = box_coder.decode(
            loc_preds.data.squeeze().cpu(),
            F.softmax(cls_preds.squeeze(), dim=1).data.cpu(),
            nms_thresh=1.0, score_thresh=0.22)
        boxes = [box for i, box in enumerate(boxes) if labels[i] == cls_id]
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            stats['x_min'] = x1
            stats['y_min'] = y1
            stats['x_max'] = x2
            stats['y_max'] = y2
            stats['score'] = scores[i]
            stats['timestamp'] = time()
            save_stats(sqlite_path, stats)

    log_prefix = get_log_prefix()
    tqdm_epochs = trange(start_epoch, start_epoch+NUM_EPOCHS, desc="Epoch",
                         ncols=0)
    for epoch in tqdm_epochs:
        trn_loss = train(epoch)
        val_loss, stats = validate(epoch, log_prefix, RUN_NAME, tqdm_epochs,
                                   trn_loss)
    avg_prec = evaluate(net, img_dir, trn_labels_fpath, IMG_SIZE,
                        args.test_code)['ap'][0]
    print("Average precision, class 0:", avg_prec)
    stats['avg_prec'] = avg_prec
    stats['timestamp'] = time()
    save_stats(sqlite_path, stats)
    if args.test_code:
        cmd = "SELECT * FROM trials WHERE trial_id = -1"
        conn = sqlite3.connect(sqlite_path)
        print(pd.read_sql_query(cmd, conn).to_string())
        for fpath in glob("checkpoints/trial--001*"):
            os.remove(fpath)
        if False:
            cmd = "DELETE FROM trials WHERE trial_id = -1"
            connect_and_execute(sqlite_path, cmd)
