#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_drq_without_pooling_average_2_flip_rot
seed=5

echo "start running $tag with seed $seed"
python train.py task=reacher_hard pooling=false aug_K=2 data_aug=flip_rot num_train_frames=1000000 experiment=$tag seed=$seed replay_buffer_num_workers=0
