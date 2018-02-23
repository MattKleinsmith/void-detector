import subprocess


def test_evaluate():
    cmd = "python evaluate.py --test-code"
    completed = subprocess.run(cmd.split())
    assert completed.returncode == 0
