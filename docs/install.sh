#!/bin/sh

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=$(pwd) # optional, defaults to ~/micromamba

eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate
micromamba create -n _psiflow_env -y python=3.10 pip ndcctools=7.14.0 -c conda-forge
micromamba activate _psiflow_env
pip install git+https://github.com/molmod/psiflow.git@v4.0.0

# create activate.sh
echo 'ORIGDIR=$PWD' >>activate.sh # prevent variable substitution
echo "cd $(pwd)" >>activate.sh
echo "export MAMBA_ROOT_PREFIX=$(pwd)" >>activate.sh
echo 'eval "$(./bin/micromamba shell hook -s posix)"' >>activate.sh
echo "micromamba activate _psiflow_env" >>activate.sh
echo 'cd $ORIGDIR' >>activate.sh # prevent variable substitution
