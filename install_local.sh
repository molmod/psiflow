# Create a psiflow installation including ModelTraining and ModelEvaluation dependencies

ENV_NAME="psiflow-dev"
micromamba env create -n $ENV_NAME python=3.14 -c conda-forge
micromamba activate $ENV_NAME

# install workqueue
micromamba install ndcctools -c conda-forge

# install psiflow
pip install -e psiflow[dev]

# install PyTorch and MACE -- CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install mace-torch
pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12

# install PyTorch and MACE -- ROCM
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
pip install mace-torch
pip install openequivariance
python -c "import openequivariance"  # compile some binary, maybe

# install basic PLUMED and python API
micromamba install plumed -c conda-forge
pip install plumed

# make PLUMED visible to the py-plumed interface
PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
echo "{\"env_vars\": {\"PLUMED_KERNEL\": \"$PLUMED_KERNEL\"}}" >> "$CONDA_PREFIX/conda-meta/state"

# install simple-dftd3
micromamba install simple-dftd3 dftd3-python -c conda-forge

# install cp2k-input-tools 
# (its dependencies are a right mess, so we fix some and pip will complain)
pip install cp2k-input-tools
pip install --force-reinstall Pint
