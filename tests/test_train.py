import subprocess


def test_train():
    cmd = "python train.py --gpu 1 --test-code"
    completed = subprocess.run(cmd.split())
    assert completed.returncode == 0
