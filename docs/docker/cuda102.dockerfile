FROM nvidia/cuda:10.2-devel-ubuntu18.04
MAINTAINER deepspeed <gqwang@baai.ac.cn>
##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Install Basic Utilities
##############################################################################
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN  sed -i s@/security.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list 
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen jq psmisc dnsutils lsof musl-dev systemd


##############################################################################
# Install Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version


##############################################################################
# Install Mellanox OFED
# https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.1-2.5.8.0/MLNX_OFED_LINUX-5.1-2.5.8.0-ubuntu18.04-x86_64.tgz
##############################################################################
RUN apt-get install -y libnuma-dev  libnuma-dev libcap2
ENV MLNX_OFED_VERSION=5.1-2.5.8.0
COPY MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz ${STAGE_DIR}
RUN cd ${STAGE_DIR} && \
    tar xvfz MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64 && \
    PATH=/usr/bin:$PATH ./mlnxofedinstall --user-space-only --without-fw-update --umad-dev-rw --all -q && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64*
 

##############################################################################
# Install nv_peer_mem
##############################################################################
#COPY nv_peer_memory ${STAGE_DIR}/nv_peer_memory
############try for more times #################
ENV NV_PEER_MEM_VERSION=1.1
ENV NV_PEER_MEM_TAG=1.1-0
RUN git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory
RUN cd ${STAGE_DIR}/nv_peer_memory && \
    ./build_module.sh && \
    cd ${STAGE_DIR} && \
    tar xzf ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
    cd ${STAGE_DIR}/nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
    apt-get update && \
    apt-get install -y dkms && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb



##############################################################################
# Install libevent && OPENMPI
# https://www.open-mpi.org/software/ompi/v4.0/
##############################################################################
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.5
COPY openmpi-4.0.5.tar.gz  ${STAGE_DIR}
COPY libevent-2.0.22-stable.tar.gz  ${STAGE_DIR}
RUN cd ${STAGE_DIR} && \
    tar zxvf libevent-2.0.22-stable.tar.gz && \
    cd libevent-2.0.22-stable && \
    ./configure --prefix=/usr && \
    make && make install
RUN cd ${STAGE_DIR} && \
    tar --no-same-owner -xzf openmpi-4.0.5.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install  && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun


##############################################################################
# Python
##############################################################################
ARG PYTHON_VERSION=3.8
RUN curl -o ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing 



##############################################################################
# Install magma-cuda
##############################################################################
COPY magma-cuda102-2.5.2-1.tar.bz2   ${STAGE_DIR}   
RUN  cd ${STAGE_DIR} && \
     /opt/conda/bin/conda install -y -c pytorch --use-local magma-cuda102-2.5.2-1.tar.bz2  && \
     /opt/conda/bin/conda clean -ya
####optional#####
#RUN  /opt/conda/bin/conda install -y -c pytorch  magma-cuda102  && \
#     /opt/conda/bin/conda clean -ya     


##############################################################################
# Export path
##############################################################################
ENV PATH /opt/conda/bin:$PATH
RUN echo "export PATH=/opt/conda/bin:\$PATH" >> /root/.bashrc
RUN pip install --upgrade pip setuptools
RUN wget https://tuna.moe/oh-my-tuna/oh-my-tuna.py && python oh-my-tuna.py


##############################################################################
# Install some Packages
##############################################################################
RUN pip install psutil \
                yappi \
                cffi \
                ipdb \
                h5py \
                pandas \
                matplotlib \
                py3nvml \
                pyarrow \
                graphviz \
                astor \
                boto3 \
                tqdm \
                sentencepiece \
                msgpack \
                requests \
                pandas \
                sphinx \
                sphinx_rtd_theme \
                sklearn \
                scikit-learn \
                nvidia-ml-py3 \
                nltk \
                rouge \
                filelock \
                fasttext \
                rouge_score \
                cupy-cuda102\
                setuptools==60.0.3


##############################################################################
# Install mpi4py
##############################################################################
RUN apt-get update && \
    apt-get install -y mpich
COPY  mpi4py-3.1.3.tar.gz ${STAGE_DIR}
RUN  cd  ${STAGE_DIR} && tar zxvf mpi4py-3.1.3.tar.gz && \
     cd mpi4py-3.1.3 &&\
     python setup.py build && python setup.py install 


##############################################################################
# PyTorch
#  clone (may be time out because of the network problem)
#RUN git clone --recursive https://github.com/pytorch/pytorch --branch v1.8.1 /opt/pytorch
#RUN cd /opt/pytorch && git checkout -f v1.8.1 && \
#    git submodule sync && git submodule update -f --init --recursive
##############################################################################
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"

COPY opt/pytorch /opt/pytorch
ENV NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu
ENV NCCL_INCLUDE_DIR=/usr/include
RUN cd /opt/pytorch && TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" USE_SYSTEM_NCCL=1 \
    pip install -v . && rm -rf /opt/pytorch 

##############################################################################
# Install vision
#RUN git clone https://github.com/pytorch/vision.git /opt/vision
##############################################################################
COPY vision /opt/vision
RUN cd /opt/vision  && pip install -v . && rm -rf /opt/vision
ENV TENSORBOARDX_VERSION=1.8
RUN pip install tensorboardX==${TENSORBOARDX_VERSION}



##############################################################################
#  Install apex
##############################################################################
#RUN git clone https://github.com/NVIDIA/apex ${STAGE_DIR}/apex
COPY apex ${STAGE_DIR}/apex
RUN cd ${STAGE_DIR}/apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && rm -rf ${STAGE_DIR}/apex



##############################################################################
# Install deepSpeed
##############################################################################
RUN pip install  py-cpuinfo
RUN apt-get install -y libaio-dev 
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
#COPY DeepSpeed ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed &&  \
    git checkout . && \
    DS_BUILD_OPS=1 ./install.sh -r
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__)"




##############################################################################
# Set SSH Config
##############################################################################


ARG SSH_PORT=6001
# RUN echo 'root:NdjeS+-4gEPmq}D' | chpasswd
# Client Liveness & Uncomment Port 22 for SSH Daemon
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN mkdir -p /var/run/sshd && cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
    sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
    sed "0,/^Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
EXPOSE ${SSH_PORT}
# Set SSH KEY
RUN printf "#StrictHostKeyChecking no\n#UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
 ssh-keygen -t rsa -f ~/.ssh/id_rsa -N "" && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
   chmod og-wx ~/.ssh/authorized_keys

