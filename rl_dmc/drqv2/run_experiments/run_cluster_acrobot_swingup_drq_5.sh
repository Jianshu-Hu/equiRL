#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_drq_shift
seed=5

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup data_aug=default num_train_frames=1100000 experiment=$tag seed=$seed replay_buffer_num_workers=0
