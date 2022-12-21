import subprocess


def exec_cmd(cmd: str, wait: bool = False) -> None:
    p = subprocess.Popen(cmd, shell=True)
    if wait:
        assert p.wait() == 0
