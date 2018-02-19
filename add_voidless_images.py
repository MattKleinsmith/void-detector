'''
Warning: This script can be tricky. Please read below:

This script assumes that all unlabeled images with an ID less than the max ID
of the labeled images are voidless images. It grabs image IDs from the image
directory. If you've added unlabeled images with an ID less than the max ID
of the labeled images, then make sure to label them before running this script.

A use case: You sampled images from the video at one frame per second, labeled
some in order, starting from the first one, and now want to add the voidless
images.

TODO: Add a "voidless" button to the hand labeler to replace this script.
'''

import os.path as osp
from glob import glob


def fname2id(fname):
    return int(fname.split('.')[0].split('_')[-1])


def line2id(line):
    fname = line.split()[0]
    return fname2id(fname)


def fpath2id(fpath):
    fname = fpath.split('/')[-1]
    return fname2id(fname)


VIDEO_ID = '20180215_185312'
# VIDEO_ID = '20180215_190227'
RAW_DATA_DIR = '../../data/voids'
LABEL_DIR = "labels"
CLASS_ID = -1  # voidless
HEIGHT = 640
WIDTH = 480

ground_truth_txt = osp.join(LABEL_DIR, VIDEO_ID + ".txt")
with open(ground_truth_txt) as f:
    ground_truth_list = f.read().splitlines()

labeled_ids = [line2id(line) for line in ground_truth_list]
max_labeled_id = max(labeled_ids)
image_dir = osp.join(RAW_DATA_DIR, VIDEO_ID)
all_ids = [fpath2id(fpath) for fpath in glob(image_dir + "/*.jpg")]
voidless_ids = [i for i in all_ids
                if i < max_labeled_id and i not in labeled_ids]
voidless_label = "1 1 {} {} {}".format(HEIGHT+1, WIDTH+1, CLASS_ID)
lines = []
for i in voidless_ids:
    fname = str(VIDEO_ID) + "_%06d.jpg" % i
    line = fname + " " + voidless_label
    lines.append(line)
ground_truth_list += lines
ground_truth_list = sorted(ground_truth_list, key=line2id)

with open(ground_truth_txt, 'w') as f:
    for line in ground_truth_list:
        f.write(line + '\n')
