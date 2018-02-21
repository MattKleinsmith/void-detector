'''Run from project root, or whichever dir that contains the labels dir'''

import os.path as osp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--video_id', default='20180215_190227', type=str, help="Base name of output. Will result in <name>.txt.")  # noqa
args = parser.parse_args()

LABEL_DIR = "labels"
ground_truth_txt = osp.join(LABEL_DIR, args.video_id + ".txt")
with open(ground_truth_txt) as f:
    ground_truth_list = f.readlines()

num_gt_voids = 0
for line in ground_truth_list:
    values = line.split()
    fname, gt = values[0], values[1:]
    # gt: list of: xmin ymin xmax ymax class
    gt_boxes = [list(map(int, gt[i*5:(i+1)*5][:-1]))
                for i in range(len(gt)//5)]
    num_gt_voids += len(gt_boxes)

print("Number of images with ground truth voids:", len(ground_truth_list))
print("Number of ground truth voids:", num_gt_voids)
