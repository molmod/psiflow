#!/bin/sh

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=$(pwd)  # optional, defaults to ~/micromamba

eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate
micromamba create -n _psiflow_env -y python=3.10 pip ndcctools -c conda-forge
micromamba activate _psiflow_env
pip install git+https://github.com/molmod/psiflow.git@ipi

# create activate.sh
echo "export ORIGDIR=$(pwd)" >> activate.sh
echo "cd $(pwd)" >> activate.sh
echo "export MAMBA_ROOT_PREFIX=$(pwd)" >> activate.sh
echo 'eval "$(./bin/micromamba shell hook -s posix)"' >> activate.sh
echo "micromamba activate _psiflow_env" >> activate.sh
echo "cd $ORIGDIR" >> activate.sh
