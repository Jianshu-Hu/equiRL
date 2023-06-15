#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_drq_without_pooling_flip
seed=4

echo "start running $tag with seed $seed"
python train.py task=reacher_hard pooling=false data_aug=flip num_train_frames=1000000 experiment=$tag seed=$seed replay_buffer_num_workers=0
