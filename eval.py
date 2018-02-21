import argparse
import os.path as osp
from time import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models.ssd import SSDBoxCoder

from model import FPNSSD512_2


parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
parser.add_argument('--test-code', action='store_true', help='Use a small sample of the data.')  # noqa
parser.add_argument('--checkpoint', default='checkpoints/2018-02-16_first-model.pth', type=str, help='Checkpoint path')  # noqa
parser.add_argument('--video-id', default='20180215_185312', type=str, choices=['20180215_185312', '20180215_190227'])  # noqa
args = parser.parse_args()

IMG_SIZE = 512
IMAGE_DIR = '../../data/voids'
LABEL_DIR = "../void-detector/labels"
VOIDS_ONLY = False
voids = "_voids" if VOIDS_ONLY else ''

img_dir = osp.join(IMAGE_DIR, args.video_id)
list_file = osp.join(LABEL_DIR, args.video_id + voids + '.txt')

print('Loading model..')
net = FPNSSD512_2()
net.load_state_dict(torch.load(args.checkpoint)['net'])

with torch.cuda.device(args.gpu):
    net.cuda()
    net.eval()

    print('Preparing dataset..')

    def transform(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(IMG_SIZE, IMG_SIZE))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, boxes, labels

    dataset = ListDataset(root=img_dir,
                          list_file=list_file,
                          transform=transform)
    if args.test_code:
        dataset.num_imgs = 10
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=2)

    box_coder = SSDBoxCoder(net)

    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []

    def eval(net, dataset):
        for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
            print('%d/%d' % (i, len(dataloader)))
            gt_boxes.append(box_targets.squeeze(0))
            gt_labels.append(label_targets.squeeze(0))

            loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
            box_preds, label_preds, score_preds = box_coder.decode(
                loc_preds.cpu().data.squeeze(),
                F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                score_thresh=0.01)

            pred_boxes.append(box_preds)
            pred_labels.append(label_preds)
            pred_scores.append(score_preds)

        ap_map_dict = voc_eval(pred_boxes, pred_labels, pred_scores, gt_boxes,
                               gt_labels, iou_thresh=0.5, use_07_metric=True)
        print(ap_map_dict)

    start = time()
    eval(net, dataset)
    print("Minutes elapsed:", (time() - start)/60)
