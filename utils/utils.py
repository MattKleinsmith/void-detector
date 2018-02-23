def videoid2videoname(i):
    id2name = {-1: '',
               0: '20180215_185312',
               1: '20180215_190227'}
    return id2name[i]

# -----------------------------------------------------------------------------


def git_hash():
    import shlex
    import subprocess
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash.decode("utf-8")


def get_datetime(tz='America/Los_Angeles', tformat="%Y-%m-%d_%H-%M-%S"):
    from datetime import datetime
    import pytz
    timezone = pytz.timezone(tz)
    now = datetime.now(timezone)
    return now.strftime(tformat)


def get_log_prefix(include_date=True):
    fpath = ''
    if include_date:
        fpath += get_datetime() + "_"
    fpath += "GIT-%s" % git_hash()
    return fpath


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
