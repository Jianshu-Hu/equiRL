#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_drq_flip_rot
seed=5

echo "start running $tag with seed $seed"
python train.py task=reacher_hard data_aug=flip_rot experiment=$tag seed=$seed replay_buffer_num_workers=1
