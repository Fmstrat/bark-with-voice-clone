FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=UTC
ARG MINICONDA_VERSION=23.1.0-1
ARG PYTHON_VERSION=3.11
ARG UID=1000
ARG GID=1000

# TZ
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Prereqs
RUN apt-get update
RUN apt-get install -y \
    curl \
    wget \
    git \
    ffmpeg \
    p7zip-full \
    gcc \
    g++ \
    vim

# User
RUN groupadd --gid $GID user
RUN useradd --no-log-init --create-home --shell /bin/bash --uid $UID --gid $GID user
USER user
ENV HOME=/home/user
WORKDIR $HOME
RUN mkdir $HOME/.cache $HOME/.config && chmod -R 777 $HOME

# Python
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh
RUN chmod +x Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh
RUN ./Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh -b -p /home/user/miniconda
ENV PATH="$HOME/miniconda/bin:$PATH"
RUN conda init
RUN conda install python=$PYTHON_VERSION
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install ipython
RUN pip3 install nbconvert
RUN pip3 install git+https://github.com/suno-ai/bark.git
RUN pip3 install git+https://github.com/liyaodev/fairseq
RUN pip3 install audiolm-pytorch
RUN pip3 install tensorboardX

ADD --chown=user:user . /app
WORKDIR /app

# RUN cp LICENSE.md LICENSE
# RUN python setup.py

