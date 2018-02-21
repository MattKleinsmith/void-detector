import torch
import torch.nn as nn

from ..fpnssd import FPNSSD512


class FPNSSD512_2(FPNSSD512):

    def __init__(self, weights_path=None):
        super().__init__(num_classes=21)
        # https://drive.google.com/open?id=1yy_kUnm_hZR3uk9yLcaQSMwxVn7wApTU
        # TODO: Use https://github.com/wkentaro/gdown
        if weights_path:
            self.load_state_dict(torch.load(weights_path))
        self.num_classes = 2  # PASCAL VOC is 21
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        in_channels = 256
        num_anchors = (4, 6, 6, 6, 6, 4, 4)
        # Reset locators and classifiers
        for i in range(len(num_anchors)):
            loc_out_channels = num_anchors[i] * 4
            cls_out_channels = num_anchors[i] * self.num_classes
            self.loc_layers += [nn.Conv2d(in_channels, loc_out_channels,
                                          kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(in_channels, cls_out_channels,
                                          kernel_size=3, padding=1)]
