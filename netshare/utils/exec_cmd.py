import subprocess


def exec_cmd(cmd, wait=False):
    p = subprocess.Popen(cmd, shell=True)
    if wait:
        p.wait()
