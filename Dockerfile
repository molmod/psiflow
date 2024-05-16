# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.5.8 as micromamba

# This is the image we are going add micromaba to:
FROM cp2k/cp2k:2023.2_openmpi_generic_psmp

USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN apt-get update
RUN apt-get install ca-certificates git -y
RUN update-ca-certificates

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# modify entrypoint to also activate cp2k, but do not set up the included (lib)torch
RUN head -n -1 /usr/local/bin/_entrypoint.sh > /usr/local/bin/entry.sh
RUN sed '/torch/Id' /opt/cp2k/tools/toolchain/install/setup > /opt/cp2k/tools/toolchain/install/setup_notorch
RUN echo "source /opt/cp2k/tools/toolchain/install/setup_notorch\n" >> /usr/local/bin/entry.sh
RUN echo "export PATH=\"/opt/cp2k/exe/local:\${PATH}\"\n" >> /usr/local/bin/entry.sh

RUN echo "exec \"\$@\"" >> /usr/local/bin/entry.sh
RUN chmod +x /usr/local/bin/entry.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/entry.sh"]
# Optional: if you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

RUN micromamba install -n base --yes -c conda-forge \
    python=3.10 pip ndcctools py-plumed && \
    micromamba clean -af --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

RUN pip install --no-cache-dir wandb plotly plumed 
RUN pip install --no-cache-dir git+https://github.com/lab-cosmo/i-pi.git@feat/socket_prefix
ARG GPU_LIBRARY
RUN pip install --no-cache-dir torch==2.1 --index-url https://download.pytorch.org/whl/${GPU_LIBRARY}
RUN pip install --no-cache-dir git+https://github.com/acesuit/mace.git@v0.3.3
ARG GIT_COMMIT_SHA
RUN pip install --no-cache-dir git+https://github.com/molmod/psiflow.git@ipi


# install GPAW with MPI from cp2k
#USER root
#RUN git clone https://gitlab.com/libxc/libxc.git &&  \
#cd libxc &&  \
#apt install autogen autoconf libtool && \
#autoreconf -i && \
#./configure && \
#make && \
#make check && \
#make install
USER root
RUN apt install libxc-dev -y
USER $MAMBA_USER
RUN source /opt/cp2k/tools/toolchain/install/setup && \
export C_INCLUDE_PATH=$CPATH && \
export LIBRARY_PATH=$LD_LIBRARY_PATH && \
echo "C INCLUDE PATH" $C_INCLUDE_PATH && \
pip install gpaw
USER root
RUN mkdir /opt/gpaw-data
RUN yes | gpaw install-data /opt/gpaw-data || true
ENV GPAW_SETUP_PATH=/opt/gpaw-data
USER $MAMBA_USER

#RUN echo "export GPAW_SETUP_PATH=\"/opt/gpaw-data\n" >> /usr/local/bin/entry.sh

#ENV C_INCLUDE_PATH=$CPATH
#RUN echo "export C_INCLUDE_PATH=\${CPATH}\"\n" >> /usr/local/bin/entry.sh
#RUN /usr/local/bin/entry.sh
#RUN echo "source /opt/cp2k/tools/toolchain/install/setup_notorch\n" >> /usr/local/bin/entry_gpaw.sh
#RUN echo "source /opt/cp2k/tools/toolchain/install/setup_notorch\n" >> /usr/local/bin/entry_gpaw.sh
