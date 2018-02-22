import hashlib
import os
import os.path as osp

try:
    import gdown2  # e.g. when running "python utils/download_model.py"
except ModuleNotFoundError:
    from utils import gdown2  # e.g. when using "from utils.download_model import download_model"


def md5sum(filename, blocksize=65536):
    # Reference: https://github.com/wkentaro/fcn/blob/de12573ee785aee3de7c051d2310aed15c62e64c/fcn/data.py#L10
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            hash.update(block)
    return hash.hexdigest()


def cached_download(url, path, md5=None, quiet=False):
    # Reference: https://github.com/wkentaro/fcn/blob/de12573ee785aee3de7c051d2310aed15c62e64c/fcn/data.py#L18

    def check_md5(path, md5):
        print('[{:s}] Checking md5 ({:s})'.format(path, md5))
        return md5sum(path) == md5

    if osp.exists(path) and not md5:
        print('[{:s}] File exists ({:s})'.format(path, md5sum(path)))
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        dirpath = osp.dirname(path)
        if not osp.exists(dirpath):
            os.makedirs(dirpath)
        return gdown2.download(url, path, quiet=quiet)


def download_model():
    url = 'https://drive.google.com/uc?id=10CUc12hH3Us9jQGm5I4x8BiszXpBbP7V'
    path = 'checkpoints/2018-02-16_first-model.pth'
    md5 = '4ed004f873e43c5f4bb75dfa8cf13891'
    cached_download(url, path, md5)


if __name__ == '__main__':
    download_model()
