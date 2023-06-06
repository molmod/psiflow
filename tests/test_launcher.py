import importlib
import subprocess
import shlex

from psiflow.execution import ContainerizedLauncher


def test_launcher():
    launcher = ContainerizedLauncher(uri='oras://ghcr.io/molmod/psiflow:1.0.0-rocm5.2', enable_gpu=False)
    command = launcher('python --version', 0, 0)
    result = subprocess.run(shlex.split(command), capture_output=True)
    returned = ''.join(result.stdout.decode('utf-8').split())
    assert 'Python3' in returned
