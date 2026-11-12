# Create a psiflow installation including ModelTraining and ModelEvaluation dependencies

ENV_NAME="psiflow-dev"
micromamba env create -n $ENV_NAME python=3.12 -c conda-forge
micromamba activate $ENV_NAME

# install workqueue
micromamba install ndcctools -c conda-forge

# install psiflow
pip install -e psiflow[dev]

# install PyTorch and MACE -- CUDA
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install mace-torch[cueq,cueq-cuda-13]

# install PyTorch and MACE -- ROCM
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/rocm6.3
pip install mace-torch[oeq]
python -c "import openequivariance"  # compile some binary, maybe

# install basic PLUMED and python API
micromamba install plumed -c conda-forge
pip install plumed

# make PLUMED visible to the py-plumed interface
export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
echo "{\"env_vars\": {\"PLUMED_KERNEL\": \"$PLUMED_KERNEL\"}}" >> "$CONDA_PREFIX/conda-meta/state"

# install simple-dftd3
micromamba install simple-dftd3 dftd3-python -c conda-forge

