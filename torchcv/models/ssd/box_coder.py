'''Encode object boxes and labels.'''
import math
import torch
import itertools

from torchcv.utils import meshgrid
from torchcv.utils.box import box_iou, box_nms, change_box_order


class SSDBoxCoder:
    def __init__(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]

                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return (i[j], j)

        # True?: default boxes are also known as "anchors" in some contexts
        # Or is an anchor a default position from which multiple default boxes
        # are formed?
        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()

        # Match ground truth boxes with default boxes based on IoU
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        # Assign ground truth boxes to unmatched default boxes if the overlap is good enough.
        # Consequence: Some ground truth boxes are matched with multiple default boxes.
        # Clarification: Each default box can have at most one ground truth box matched with
        # it. Some default boxes will not be matched with a ground truth box.
        mask = (index<0) & (ious.max(1)[0]>=0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        # Shape: (num_default_boxes, 4)
        # Each default box index is replaced with a ground truth box that it
        # was matched with. Unmatched default boxes are given the first ground
        # truth box, but this won't affect the location loss since unmatched
        # default boxes are tracked as "negative examples" via an index of -1.
        # Later, all class labels will be incremented, leaving the class label
        # of 0 free for new use. This is the class label we will assign to
        # negative examples, which are those with an index of -1.
        # I'm not sure why we couldn't just give negative examples a class
        # label of -1 and not change the original ground truth class labels.
        boxes = boxes[index.clamp(min=0)]  # negative index not supported

        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:, :2]-default_boxes[:, :2]) / default_boxes[:, 2:] / variances[0]
        loc_wh = torch.log(boxes[:, 2:]/default_boxes[:, 2:]) / variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)

        # Add one to the label ID of each default box that was matched with a
        # ground truth box. Reason: We must make room for the "unassigned"
        # class. F.cross_entropy doesn't allow negative class numbers,
        # so we can use -1 for this class. Not sure why we don't use the next
        # available positive number, but this works.
        cls_targets = 1 + labels[index.clamp(min=0)]  # Positive examples
        # Assign a class ID of 0 to unmatched default boxes. These will be
        # considered negative examples in the location loss function.
        # See SSDLoss
        cls_targets[index < 0] = 0  # Negative examples

        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        variances = (0.1, 0.2)
        xy = loc_preds[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:]*variances[1]) * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:, i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]
            if nms_thresh < 1.0:
                keep = box_nms(box, score, nms_thresh)
            else:
                keep = list(range(len(box)))
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        try:
            boxes = torch.cat(boxes, 0)
        except RuntimeError:
            return torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([])
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores
