'''
Running this script requires nvidia-docker 2.0.

If you want to test code changes with this script, you'll have to rebuild the
Docker image. I recommend using checkpoint2drawings.py and other scripts
directly if you want to test code changes.

# TODO: Allow non-Docker
'''

import argparse
import subprocess
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('images', type=str, help='Directory of images to run the model on')  # noqa
parser.add_argument('--out-dir', type=str, default='', help='Directory to write predictions to')  # noqa
parser.add_argument('--from-docker', action='store_true', help='Requires nvidia-docker 2.0')  # noqa
args = parser.parse_args()

if not args.from_docker:
    in_dir = osp.abspath(args.images)
    out_dir = (osp.abspath(args.out_dir) if args.out_dir else
               osp.join(in_dir, "void-detector-outputs"))
    print("Host directories:\n{2}{0}\n{2}{1}".format(in_dir, out_dir, ' '*4))
    cmd = "docker run --rm --runtime=nvidia --ipc=host"
    cmd += " -v {}:/inputs -v {}:/outputs".format(in_dir, out_dir)
    cmd += " matthewkleinsmith/void-detector"
else:
    cmd = "ipython -- utils/checkpoint2drawings.py"
    cmd += " --input /inputs --output /outputs"
subprocess.run(cmd.split())
