import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _hard_negative_mining(self, cls_loss, pos, ratio=3, pure=5):
        '''
           Order the negative examples by decreasing loss, and return the
           first N of them, where N is three times the number of positive
           examples. If there are no positive examples, then N equals 5.

           This is a special form of class weighting to deal with unbalanced
           classes.

           This function prevents the negative examples from overwhelming the
           positive examples (via their great number), while also focusing on
           the most difficult negative examples. Without this function or a
           similar one, the model would likely label every example as negative.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].
          ratio: Ratio of negative examples to positive examples. This is a hyperparameter.
          pure: Number of negative examples to return when there are no positive examples. This is a hyperparameter.

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses. Consequence: pays attention to largest losses first
        _, rank = idx.sort(1)      # [N,#anchors]

        if pos.data.long().sum() == 0:
            num_neg = pure + pos.long().sum(1)  # [N,]
        else:
            num_neg = ratio * pos.long().sum(1)  # [N,]
        neg = rank < num_neg[:, None]   # [N, #anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        # Only consider voids to be positive examples.
        # Consequence: Balancing: Don't use every non-void region, since doing
        # so would overwhelm the void regions, since there are many non-void
        # regions. This balancing happens via hard-negative mining.
        pos = cls_targets > 0  # [N,#anchors]

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        # Ignore negative examples when calculating location loss
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)
        num_pos = pos.data.long().sum()
        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        batch_size, num_boxes, num_classes = cls_preds.size()
        cls_loss = F.cross_entropy(cls_preds.view(-1, num_classes),
                                   cls_targets.view(-1), reduce=False)  # [N*#anchors,]
        cls_loss = cls_loss.view(batch_size, -1)
        print("cls_loss.view", cls_loss)
        cls_loss[cls_targets < 0] = 0  # set ignored loss to 0  # Not sure when a class label would be less than one.
        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos | neg].sum()
        num_neg = neg.data.long().sum()
        if num_pos:  # I'd prefer to divide by num_examples, but I'm minimizing the code changes for now.
            cls_loss.data /= num_pos
            loc_loss.data /= num_pos
        else:
            cls_loss.data /= num_neg

        loss = loc_loss + cls_loss
        print('loc_loss: {:.3f} | cls_loss: {:.3f} | loss: {:.3f}'.format(
            loc_loss.data[0], cls_loss.data[0], loss.data[0]))
        return loss
