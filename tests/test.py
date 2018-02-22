import subprocess


def test_train():
    cmd = "python train.py --gpu 1 --test-code"
    completed = subprocess.run(cmd.split())
    assert completed.returncode == 0


def test_evaluate():
    cmd = "python evaluate.py --test-code"
    completed = subprocess.run(cmd.split())
    assert completed.returncode == 0


def test_demo():
    cmd = "python demo.py outputs/20180215_190227 --dockerless --test-code"
    completed = subprocess.run(cmd.split())
    assert completed.returncode == 0
