import argparse
import subprocess
import os.path as osp

from utils.download_model import download_model

parser = argparse.ArgumentParser()
parser.add_argument('images', type=str, help='Directory of images to run the model on')  # noqa
parser.add_argument('--out-dir', type=str, default='', help='Directory to write predictions to')  # noqa
parser.add_argument('--dockerless', action='store_true', help='Run without Docker')  # noqa
parser.add_argument('--gpu', default='0', type=int, help='GPU ID (nvidia-smi)')  # noqa
args = parser.parse_args()

in_dir = osp.abspath(args.images)
out_dir = (osp.abspath(args.out_dir) if args.out_dir else
           osp.join(in_dir, "void-detector-outputs"))
values = [in_dir, out_dir, args.gpu]

if args.dockerless:
    download_model()
    cmd = "ipython -- utils/checkpoint2drawings.py"
    cmd += " --input {} --output {} --gpu {}".format(*values)
else:
    print("Host directories:\n{2}{0}\n{2}{1}".format(in_dir, out_dir, ' '*4))
    cmd = "docker run --rm -it --runtime=nvidia --ipc=host"
    cmd += " -v {}:/inputs -v {}:/outputs -e GPU_ID={}".format(*values)
    cmd += " matthewkleinsmith/void-detector"
subprocess.run(cmd.split())
