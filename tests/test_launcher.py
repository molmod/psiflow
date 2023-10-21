import importlib
import subprocess
import shlex
import pytest

from psiflow.parsl_utils import ContainerizedLauncher


def get_apptainer_or_singularity():
    if (
        subprocess.run(["which apptainer"], shell=True, capture_output=True).stdout
        is not None
    ):
        apptainer_or_singularity = "apptainer"
    elif (
        subprocess.run(["which singularity"], shell=True, capture_output=True).stdout
        is not None
    ):
        apptainer_or_singularity = "singularity"
    else:
        apptainer_or_singularity = None
    return apptainer_or_singularity


# @pytest.mark.skipif(get_apptainer_or_singularity() is None, reason='no apptainer/singularity found')
@pytest.mark.skip
def test_launcher():
    launcher = ContainerizedLauncher(
        uri="oras://ghcr.io/molmod/psiflow:1.0.0-rocm5.2",
        enable_gpu=False,
        apptainer_or_singularity=get_apptainer_or_singularity(),
    )
    command = launcher("python --version", 0, 0)
    result = subprocess.run(shlex.split(command), capture_output=True)
    returned = "".join(result.stdout.decode("utf-8").split())
    assert "Python3" in returned
