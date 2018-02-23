import subprocess


def test_demo():
    cmd = "python demo.py outputs/20180215_190227 --dockerless --test-code"
    completed = subprocess.run(cmd.split())
    assert completed.returncode == 0
