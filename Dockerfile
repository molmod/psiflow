# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.4.2 as micromamba

# This is the image we are going add micromaba to:
FROM svandenhaute/cp2k:2023.1
#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# modify entrypoint to also activate cp2k 
RUN head -n -1 /usr/local/bin/_entrypoint.sh > /usr/local/bin/entry.sh
RUN echo "source /opt/cp2k-toolchain/install/setup\n" >> /usr/local/bin/entry.sh
RUN echo "export PATH=\"/opt/cp2k/exe/local:\${PATH}\"\n" >> /usr/local/bin/entry.sh
RUN echo "export PATH=\"/opt/cp2k-toolchain/install/mpich-4.0.3/bin:\${PATH}\"\n" >> /usr/local/bin/entry.sh
RUN echo "exec \"\$@\"" >> /usr/local/bin/entry.sh
RUN chmod +x /usr/local/bin/entry.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

RUN micromamba install --yes --name base --channel conda-forge \
      jq && \
    micromamba clean --all --yes

RUN CONDA_OVERRIDE_CUDA="11.8" micromamba install -n base --yes -c conda-forge \
    python=3.9 pip ndcctools=7.6.1 \
    openmm-plumed openmm-torch pytorch=1.13.1=cuda* \
    nwchem py-plumed && \
    micromamba clean -af --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

RUN pip install cython==0.29.36 matscipy prettytable && \
    pip install git+https://github.com/molmod/molmod && \
    pip install git+https://github.com/molmod/yaff && \
    pip install e3nn==0.4.4
RUN pip install numpy ase tqdm pyyaml 'torch-runstats>=0.2.0' 'torch-ema>=0.3.0' mdtraj tables
RUN pip install git+https://github.com/acesuit/MACE.git@55f7411 && \
    pip install git+https://github.com/mir-group/nequip.git@develop --no-deps && \
    pip install git+https://github.com/mir-group/allegro --no-deps && \
    pip install git+https://github.com/svandenhaute/openmm-ml.git@triclinic

ARG GIT_COMMIT_SHA
RUN pip install git+https://github.com/molmod/psiflow
RUN pip cache purge

ENV OMPI_MCA_plm_rsh_agent=
ENV OMP_PROC_BIND=TRUE

ENTRYPOINT ["/usr/local/bin/entry.sh"]
