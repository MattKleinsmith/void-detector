FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# To allow unicode characters in the terminal
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# To make the CUDA device order match the IDs in nvidia-smi
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

##############
## Anaconda ##
##############

# https://github.com/ContinuumIO/docker-images/blob/master/anaconda3/Dockerfile

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git \
    vim && \
    rm -rf /var/lib/apt/lists/* && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

# Tini is unneeded because Docker 1.13 has the --init flag.

ENV PATH /opt/conda/bin:$PATH

########################
## Intel MKL settings ##
########################

# To allow setting the number of CPU threads
RUN conda install mkl-service

# To allow MKL to see all CPU threads
ENV MKL_DYNAMIC=FALSE

############
## OpenCV ##
############

RUN pip install opencv-python

#############
## PyTorch ##
#############

RUN conda install pytorch torchvision cuda90 -c pytorch

###################
## void-detector ##
###################

RUN pip install \
    tqdm

# BitBucket allows for read-only access for private repos for free
RUN git clone https://void-detector:LzPxmG9HjBkzQfrs3Aa3@bitbucket.org/void-detector/void-detector.git

WORKDIR void-detector

RUN python utils/download_model.py

##########
## Misc ##
##########

# To display the time in bash prompts
RUN echo "PS1=\"[\D{%T}] \"$PS1" >> ~/.bashrc

ENV TEST_CODE=""

CMD ipython -- utils/checkpoint2drawings.py --input /inputs --output /outputs --gpu $GPU_ID $TEST_CODE
