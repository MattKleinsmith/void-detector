def get_log_prefix(include_date=True):
    def git_hash():
        import shlex
        import subprocess
        cmd = 'git log -n 1 --pretty="%h"'
        hash = subprocess.check_output(shlex.split(cmd)).strip()
        return hash

    def get_datetime(tz='America/Los_Angeles', tformat="%Y-%m-%d--%H-%M-%S"):
        from datetime import datetime
        import pytz
        timezone = pytz.timezone(tz)
        now = datetime.now(timezone)
        return now.strftime(tformat)

    fpath = ''
    if include_date:
        fpath += get_datetime() + "_"
    fpath += "GIT-%s" % git_hash().decode("utf-8")
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
