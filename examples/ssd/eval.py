import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models.fpnssd import FPNSSD512_2
from torchcv.models.ssd import SSDBoxCoder


VIDEO_ID = '20180215_185312'
#VIDEO_ID = '20180215_190227'
CKPT_PATH = './examples/ssd/checkpoint/200_epoch_backup.pth'
IMG_SIZE = 512

img_dir = '/data/voids/' + VIDEO_ID
list_file = '/data/voids/' + VIDEO_ID + '.txt'


print('Loading model..')
net = FPNSSD512_2()
net.load_state_dict(torch.load(CKPT_PATH)['net'])
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


eval(net, dataset)
