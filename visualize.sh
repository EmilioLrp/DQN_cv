#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cv_env
tensorboard --logdir=./ --host=localhost