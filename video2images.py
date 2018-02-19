"""
# In bash

# Back up videos
cp 20180215_185312.mp4 /media/mwk/3TB/voids/
cp 20180215_190227.mp4 /media/mwk/3TB/voids/

# Extract frames at near-original quality
ffmpeg -i 20180215_185312.mp4 -q:v 2 20180215_185312/20180215_185312_%06d.jpg
ffmpeg -i 20180215_190227.mp4 -q:v 2 20180215_190227/20180215_190227_%06d.jpg

# Back up frames
cp -r 20180215_185312 /media/mwk/3TB/voids/
cp -r 20180215_190227 /media/mwk/3TB/voids/

-------------------------------------------------------------------------------

This script: Extract one frame per second. Delete the rest.
Use this script in each directory.
Rationale:
    Hypothesis: Labeling two different voids will give more information
        than labeling the same void twice in different positions.
    Hypothesis: We can use object tracking to automatically label the
        the same void through many frames given one manual label.
Why not use FFMPEG to extract one frame per second?:
    Because I want the ID numbers between 1 FPS and 30 FPS to align, so I can
    easily attempt object tracking later.
TODO (do after getting the next batch of data):
    - Use glob('*/*.jpg') in the parent directory to avoid having to run the
      script in multiple directories.
    - Port the bash code to Python, at least via subprocess and shlex.
"""

from glob import glob
import os

fnames = glob('*')
prefix = '_'.join(fnames[0].split('_')[:2])

ids = [int(f.split('_')[-1][:-4]) for f in fnames]
one_fps = [i for i in ids if i % 30 == 0]

print("num frames:", len(fnames))
print("num extracted frames:", len(one_fps))
print("first ten IDs:", sorted(one_fps)[:10])

to_remove = [prefix + '_%06d.jpg' % i for i in ids if i not in one_fps]
for i in to_remove:
    os.remove(i)
