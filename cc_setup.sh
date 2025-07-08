#!/bin/bash

#module reset
#module load python/3.8 cuda cudnn

export ENV_TMPDIR=$(find /localscratch/$USER*)
if [ ! -d "$ENV_TMPDIR/env/" ]; then
  virtualenv $ENV_TMPDIR/env/
fi

source $ENV_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
# pip install --no-index torch torchvision tensorboard opencv-python scipy nibabel

python ./Code/Train.py
