import os
import os.path as osp
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import cv2

from torchcv.models.ssd import SSDBoxCoder

from model import FPNSSD512_2


def get_ground_truth(line):
    splited = line.strip().split()
    fname = splited[0]
    boxes = []
    labels = []
    num_boxes = (len(splited) - 1) // 5
    for i in range(num_boxes):
        xmin = int(np.round(float((splited[1+5*i])))) - 1
        ymin = int(np.round(float((splited[2+5*i])))) - 1
        xmax = int(np.round(float((splited[3+5*i])))) - 1
        ymax = int(np.round(float((splited[4+5*i])))) - 1
        c = splited[5+5*i]
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(c))
    return fname, boxes, labels


def get_pred_boxes(img, img_size, net, cls_id=0):  # 0: void
    x = img.resize((img_size, img_size))
    x = transform(x)
    x = Variable(x, volatile=True).cuda()
    loc_preds, cls_preds = net(x.unsqueeze(0))
    box_coder = SSDBoxCoder(net)
    boxes, labels, scores = box_coder.decode(
        loc_preds.data.squeeze().cpu(),
        F.softmax(cls_preds.squeeze(), dim=1).data.cpu())
    boxes = [box for i, box in enumerate(boxes) if labels[i] == cls_id]
    return boxes


def draw_preds_and_save(img, img_size, boxes, out_dir, fname):
    shape = (img_size, img_size)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)
    for box in boxes:
        box = list(np.int64(np.round(box)))
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(osp.join(out_dir, fname), img)


# VIDEO_ID = "20180215_185312"
VIDEO_ID = "20180215_190227"
OUTPUT_SUFFIX = ''
USE_GROUND_TRUTH = True
IMG_SIZE = 512
RAW_DATA_DIR = "../../data/voids"
LABEL_DIR = "labels"
CKPT_NAME = "200_epoch_backup.pth"
CLS_ID = 0  # void

use_gt = "_gt" if USE_GROUND_TRUTH else ''
suffix = "_" + OUTPUT_SUFFIX if OUTPUT_SUFFIX else ''
ckpt_path = osp.join("checkpoints", CKPT_NAME)
in_dir = osp.join(RAW_DATA_DIR, VIDEO_ID)
out_dir = osp.join("outputs", VIDEO_ID + use_gt + suffix)
os.makedirs(out_dir, exist_ok=True)

print('Loading model..')
net = FPNSSD512_2()
ckpt = torch.load(ckpt_path)
net.load_state_dict(ckpt['net'])
net.cuda()
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

if USE_GROUND_TRUTH:
    ground_truth_txt = osp.join(LABEL_DIR, VIDEO_ID + ".txt")
    with open(ground_truth_txt) as f:
        ground_truth_lines = f.readlines()
    for line in ground_truth_lines:
        fname, gt_boxes, gt_labels = get_ground_truth(line)
        gt_boxes = [box for i, box in enumerate(gt_boxes)
                    if gt_labels[i] == CLS_ID]
        print(fname)
        img = Image.open(osp.join(in_dir, fname))
        pred_boxes = get_pred_boxes(img, IMG_SIZE, net, cls_id=CLS_ID)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_preds_and_save(img, IMG_SIZE, pred_boxes, out_dir, fname)
else:
    fpaths = glob(in_dir + "/*.jpg")
    for fpath in fpaths:
        fname = fpath.split('/')[-1]
        print(fname)
        img = Image.open(osp.join(in_dir, fname))
        pred_boxes = get_pred_boxes(img, IMG_SIZE, net, cls_id=CLS_ID)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        draw_preds_and_save(img, IMG_SIZE, pred_boxes, out_dir, fname)
