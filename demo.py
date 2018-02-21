'''Running this script requires nvidia-docker 2.0'''

# TODO: Allow non-Docker

import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, help='Directory of images to run the model on')  # noqa
parser.add_argument('--out-dir', type=str, default='', help='Directory to write predictions to')  # noqa
parser.add_argument('--run-docker', action='store_false', help='Requires nvidia-docker 2.0')  # noqa
args = parser.parse_args()

if args.run_docker:
    out_dir = args.out_dir if args.out_dir else args.images + "/outputs"
    cmd = "docker run --rm --runtime=nvidia"
    cmd += " -v {}:/inputs -v {}:/outputs".format(args.images, out_dir)
else:
    cmd = "ipython -- utils/checkpoint2drawings.py"
    cmd += " --input /inputs --output /outputs"
subprocess.run(cmd.split())
