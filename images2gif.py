from glob import glob
import os
import os.path as osp
import shutil
import subprocess

start = 1080
end = 2310
fps = 2

fnames = glob('*.jpg')
ids = [int(f.split('_')[-1][:-4]) for f in fnames]
interval = [i for i in ids if i >= start and i <= end]
prefix = '_'.join(fnames[0].split('_')[:2])
fnames = [prefix + '_%06d.jpg' % i for i in ids if i in interval]

gif_dir = 'gif'
os.makedirs(gif_dir, exist_ok=True)
for fname in fnames:
    shutil.copyfile(fname, osp.join(gif_dir, fname))

delay = 100/fps
video_id = osp.basename(os.getcwd())
values = delay, gif_dir, video_id, fps
cmd = "convert -delay {0} -loop 0 {1}/*.jpg {1}/{2}_{3}fps.gif".format(*values)
completed = subprocess.run(cmd.split())
print('returncode:', completed.returncode)
