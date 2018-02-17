"""
2018-02-16:
- The VOC PASCAL format defines the top-left corner as (1, 1), not (0, 0).
Make sure your labeler takes this into account, or add one to each coordinate
here.
- The labeler program, reasonably, stores bounding box information of name.jpg
in name.txt, with each bounding box on a separate line. I'll need to convert
this to torchcv format, where all the bounding boxes for a single image are on
one line.

Bounding box labeler: https://github.com/Cartucho/yolo-boundingbox-labeler-GUI
My fork (not on master):
https://github.com/MattKleinsmith/yolo-boundingbox-labeler-GUI
"""

from glob import glob

LABELER_ADDED_ONE_TO_EACH_COORD = True

# Run from bbox_text directory
fnames = glob("*.txt")
fnames.sort()

with open("train.txt", 'a') as trn_file:
    for fname in fnames:
        with open(fname) as f:
            content = f.readlines()
        boxes = []
        for line in content:
            values_str = line.split()
            coords_str, class_index = values_str[:-1], values_str[-1]
            if LABELER_ADDED_ONE_TO_EACH_COORD:
                coords = list(map(int, coords_str))
            else:
                coords = list(map(lambda x: x+1, map(int, coords_str)))
            box = coords + [class_index]
            boxes += box
        fname = fname.split('.txt')[0] + ".jpg"
        trn_file.write(' '.join([fname] + list(map(str, boxes))) + '\n')
