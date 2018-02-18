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

from torchcv.models.fpnssd import FPNSSD512_2
from torchcv.models.ssd import SSDBoxCoder


def get_predictions(img, img_size, net):
    x = img.resize((img_size, img_size))
    x = transform(x)
    x = Variable(x, volatile=True).cuda()
    loc_preds, cls_preds = net(x.unsqueeze(0))
    box_coder = SSDBoxCoder(net)
    boxes, labels, scores = box_coder.decode(
        loc_preds.data.squeeze().cpu(),
        F.softmax(cls_preds.squeeze(), dim=1).data.cpu())
    return boxes


def draw_preds_and_save(img, img_size, boxes, out_dir, fname):
    shape = (img_size, img_size)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)
    for box in boxes:
        box = list(np.int64(np.round(box)))
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(osp.join(out_dir, fname), img)


# video_id = '20180215_185312'
video_id = '20180215_190227'
have_gt = False

data_dir = '/data/voids'
in_dir = osp.join(data_dir, video_id)
out_dir = osp.join("../void-torchcv/outputs", video_id)
os.makedirs(out_dir, exist_ok=True)

ckpt_path = "../void-torchcv/examples/ssd/checkpoint/200_epoch_backup.pth"
base_weight_path = "../void-torchcv/examples/ssd/checkpoint/fpnssd512_20_trained.pth"
img_size = 512

print('Loading model..')
net = FPNSSD512_2(base_weight_path)
ckpt = torch.load(ckpt_path)
net.load_state_dict(ckpt['net'])
net.cuda()
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

if have_gt:
    ground_truth_txt = osp.join(data_dir, video_id + ".txt")
    with open(ground_truth_txt) as f:
        ground_truth_list = f.readlines()

    for line in ground_truth_list:
        values = line.split()
        fname, gt = values[0], values[1:]
        print(fname)

        img = Image.open(osp.join(in_dir, fname))
        boxes = get_predictions(img, img_size, net)

        # Get and draw ground truth boxes
        # gt: list of: xmin ymin xmax ymax class
        gt_boxes = [list(map(int, gt[i*5:(i+1)*5][:-1]))
                    for i in range(len(gt)//5)]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        draw_preds_and_save(img, img_size, boxes, out_dir, fname)
else:
    fpaths = glob(in_dir + "/*.jpg")
    for fpath in fpaths:
        fname = fpath.split('/')[-1]
        print(fname)

        img = Image.open(osp.join(in_dir, fname))
        boxes = get_predictions(img, img_size, net)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        draw_preds_and_save(img, img_size, boxes, out_dir, fname)
