import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.autograd import Variable

from torchcv.models.fpnssd import FPNSSD512_2
from torchcv.models.ssd import SSDBoxCoder


in_dir = '/data/voids/20180215_185312'
out_dir = "../void-torchcv/outputs/20180215_185312/"
ground_truth_txt = "/data/voids/20180215_185312.txt"
ckpt_path = "../void-torchcv/examples/ssd/checkpoint/200_epoch_backup.pth"
base_weight_path = "../void-torchcv/examples/ssd/checkpoint/fpnssd512_20_trained.pth"
img_size = 512

os.makedirs(out_dir, exist_ok=True)

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

with open(ground_truth_txt) as f:
    ground_truth_list = f.readlines()

for line in ground_truth_list:
    values = line.split()
    fname, gt = values[0], values[1:]
    print(fname)

    img = Image.open(osp.join(in_dir, fname))

    # Get predictions
    x = img.resize((img_size, img_size))
    x = transform(x)
    x = Variable(x, volatile=True).cuda()
    loc_preds, cls_preds = net(x.unsqueeze(0))
    box_coder = SSDBoxCoder(net)
    boxes, labels, scores = box_coder.decode(
        loc_preds.data.squeeze().cpu(),
        F.softmax(cls_preds.squeeze(), dim=1).data.cpu())

    # Get ground truth boxes
    # gt: list of: xmin ymin xmax ymax class
    gt_boxes = [list(map(int, gt[i*5:(i+1)*5][:-1]))
                for i in range(len(gt)//5)]

    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    for box in gt_boxes:
        draw.rectangle(box, outline='green')
    img = img.resize((img_size, img_size))
    draw = ImageDraw.Draw(img)
    for box in boxes:
        box = list(np.int64(np.round(box)))
        draw.rectangle(box, outline='red')

    # Save image with ground truth and predictions
    img.save(osp.join(out_dir, fname))
