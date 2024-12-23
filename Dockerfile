FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    cmake \
    wget \
    git \
    zlib1g-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libgsl-dev \
    perl \
    fftw3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# Install PLUMED
ARG PLUMED_VERSION
RUN wget https://github.com/plumed/plumed2/archive/refs/tags/v${PLUMED_VERSION}.tar.gz \
    && tar -xzf v${PLUMED_VERSION}.tar.gz \
    && cd plumed2-${PLUMED_VERSION} \
    && ./configure --prefix=/usr/local/plumed \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf plumed2-${PLUMED_VERSION} v${PLUMED_VERSION}.tar.gz

# Ensure cctools can find the Python environment
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages:$PYTHONPATH"
ENV PATH="/opt/venv/bin:$PATH"

# Install cctools
ARG CCTOOLS_VERSION
RUN wget https://github.com/cooperative-computing-lab/cctools/archive/refs/tags/release/${CCTOOLS_VERSION}.tar.gz \
    && tar -xzf ${CCTOOLS_VERSION}.tar.gz \
    && cd cctools-release-${CCTOOLS_VERSION} \
    && ./configure --prefix=/usr/local/cctools \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf cctools-release-${CCTOOLS_VERSION} ${CCTOOLS_VERSION}.tar.gz

# Set environment variables for PLUMED and cctools
ENV PATH="/usr/local/plumed/bin:/usr/local/cctools/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/plumed/lib:/usr/local/cctools/lib:$LD_LIBRARY_PATH"

ARG PSIFLOW_VERSION
ARG PARSL_VERSION
ARG GPU_LIBRARY
RUN /bin/bash -c -o pipefail \
    "source /opt/venv/bin/activate && \
     pip install --no-cache-dir pyfftw colorcet wandb pandas plotly plumed 'numpy<2.0.0' && \
     pip install --no-cache-dir git+https://github.com/i-pi/i-pi.git@v3.0.0-beta4 && \
     pip install --no-cache-dir torch>=2.5 --index-url https://download.pytorch.org/whl/${GPU_LIBRARY} && \
     pip install --no-cache-dir git+https://github.com/acesuit/mace.git@v0.3.5"
ARG DATE
RUN /bin/bash -c -o pipefail \
     "pip install --no-cache-dir git+https://github.com/molmod/psiflow.git@${PSIFLOW_VERSION}"

# Set entrypoint
RUN echo '#!/bin/bash' >> /opt/entry.sh && \
    echo 'source /opt/venv/bin/activate' >> /opt/entry.sh && \
    echo 'export PLUMED_KERNEL=/usr/local/plumed/lib/libplumedKernel.so' >> /opt/entry.sh && \
    echo '"$@"' >> /opt/entry.sh
RUN chmod +x /opt/entry.sh
ENTRYPOINT ["/opt/entry.sh"]

# Default command
CMD ["bash"]
