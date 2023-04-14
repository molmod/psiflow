import importlib

from psiflow.execution import ContainerizedLauncher


VERSION = importlib.metadata.version('psiflow')
print(VERSION)


def test_launcher():
    launcher = ContainerizedLauncher(enable_gpu=False)
    start = 'apptainer exec --no-eval -e --no-mount $HOME/.local -W /tmp --writable-tmpfs --bind'
    end   = 'docker://ghcr.io/svandenhaute/psiflow:' + VERSION + '-cuda11.3 /usr/local/bin/_entrypoint.sh '
    assert launcher.launch_command.startswith(start)
    assert launcher.launch_command.endswith(end)

    launcher = ContainerizedLauncher(enable_gpu=True)
    start = 'apptainer exec --no-eval -e --no-mount $HOME/.local -W /tmp --writable-tmpfs --bind'
    end   = '--nv docker://ghcr.io/svandenhaute/psiflow:' + VERSION + '-cuda11.3 /usr/local/bin/_entrypoint.sh '
    assert launcher.launch_command.startswith(start)
    assert launcher.launch_command.endswith(end)

    launcher = ContainerizedLauncher(enable_gpu=True, tag='v213-rocm5.3')
    assert '--rocm' in launcher.launch_command
    assert not '--nv' in launcher.launch_command

