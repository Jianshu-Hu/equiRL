#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_drq_without_pooling_shift
seed=1

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup pooling=false data_aug=default num_train_frames=1000000 experiment=$tag seed=$seed replay_buffer_num_workers=0
