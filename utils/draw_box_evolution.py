import os
import os.path as osp

from PIL import Image
import cv2
import numpy as np

from utils.sql import select


def draw_box_evolution(img, sqlite_path, trial_id, model_img_size):
    out_dir = "outputs/trial-{:04d}".format(trial_id)
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h0, w0 = img.shape[:2]
    h1, w1 = (model_img_size, model_img_size)
    x_trans = w0/w1
    y_trans = h0/h1
    cmd = """SELECT epoch, x_min, y_min, x_max, y_max FROM trials
             WHERE trial_id = ? AND x_min IS NOT NULL"""
    parameters = [trial_id]
    values = select(sqlite_path, cmd, parameters)
    epochs = [v[0] for v in values]
    for epoch in epochs:
        img_copy = img.copy()
        boxes = [v[1:] for v in values if v[0] == epoch]
        for x1, y1, x2, y2 in boxes:
            x1 = np.int64(np.round(x1 * x_trans))
            x2 = np.int64(np.round(x2 * x_trans))
            y1 = np.int64(np.round(y1 * y_trans))
            y2 = np.int64(np.round(y2 * y_trans))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        fpath = osp.join(out_dir, "epoch-{:03d}.jpg".format(epoch))
        cv2.imwrite(fpath, img_copy)


img = Image.open("docs/20180215_190227_002190.jpg")
sqlite_path = "database.sqlite3"
trial_id = 1
model_img_size = 512
draw_box_evolution(img, sqlite_path, trial_id, model_img_size)
