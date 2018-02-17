import os.path as osp

video_id = '20180215_185312'
# video_id = '20180215_190227'
data_dir = '/data/voids'
ground_truth_txt = osp.join(data_dir, video_id + ".txt")
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
