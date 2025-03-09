#!/bin/bash

touch ~/.no_auto_tmux

conda install -c nvidia cuda-compiler

conda env create -f environment.yml

conda init

source ~/.bashrc

conda activate smes_multimodal

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

pip uninstall bitsandbytes -y
pip install --upgrade bitsandbytes
pip install deepspeed
pip install transformers