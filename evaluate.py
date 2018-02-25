import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models.ssd import SSDBoxCoder

from torchcv.models.void_models import FPNSSD512_2
from utils import videoid2videoname


def evaluate(net, img_dir, list_file, img_size, test_code):
    net.cuda()
    net.eval()

    def transform(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(img_size, img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, boxes, labels

    print('Loading dataset..')
    dataset = ListDataset(root=img_dir, list_file=list_file,
                          transform=transform)
    if test_code:
        dataset.num_imgs = 1
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=2)

    box_coder = SSDBoxCoder(net)
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []
    tqdm_dl = tqdm(dl, desc="Evaluate", ncols=0)
    for i, (inputs, box_targets, label_targets) in enumerate(tqdm_dl):
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
                           gt_labels, iou_thresh=0.5, use_07_metric=False)
    return ap_map_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
    parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
    parser.add_argument('--test-code', action='store_true', help='Use a small sample of the data.')  # noqa
    parser.add_argument('--checkpoint', default='checkpoints/2018-02-16_first-model.pth', type=str, help='Checkpoint path')  # noqa
    parser.add_argument('--video-id', default=0, type=int, choices=[-1, 0, 1])  # noqa
    args = parser.parse_args()

    IMG_SIZE = 512
    IMAGE_DIR = '../../data/voids'
    LABEL_DIR = "../void-detector/labels"
    VOIDS_ONLY = False
    voids = "_voids" if VOIDS_ONLY else ''

    video_name = videoid2videoname(args.video_id)

    img_dir = osp.join(IMAGE_DIR, video_name)
    list_file = osp.join(LABEL_DIR, video_name + voids + '.txt')

    print('Loading model..')
    net = FPNSSD512_2()
    try:
        net.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    except KeyError:
        net.load_state_dict(torch.load(args.checkpoint)['net'])
    with torch.cuda.device(args.gpu):
        evaluate(net, img_dir, list_file, IMG_SIZE, args.test_code)
