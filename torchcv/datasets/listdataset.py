from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class ListDataset(data.Dataset):
    '''Load image/labels/boxes from a list file.

    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    PASCAL VOC annotations use (1, 1) as the top-left corner, not (0, 0).
    '''
    def __init__(self, root, list_file, transform=None, test_code=False):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            if test_code:
                lines = lines[:10]
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = float(splited[1+5*i]) - 1
                ymin = float(splited[2+5*i]) - 1
                xmax = float(splited[3+5*i]) - 1
                ymax = float(splited[4+5*i]) - 1
                c = splited[5+5*i]
                box.append([xmin, ymin, xmax, ymax])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()  # use clone to avoid any potential change.
        labels = self.labels[idx].clone()

        if self.transform:
            try:
                img, boxes, labels = self.transform(img, boxes, labels)
            except RuntimeError:
                print(fname)
                # Last error caught: NAME.jpg was in the list of files but
                # had no bounding boxes. I must have made a bounding box,
                # which made the file in the labeler directory, and then
                # deleted the bounding box, which doesn't delete the file
                # if it's empty.
                raise RuntimeError
        return img, boxes, labels

    def __len__(self):
        return self.num_imgs
