#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=cheetah_run_drq_without_pooling_shift
seed=5

echo "start running $tag with seed $seed"
python train.py task=cheetah_run pooling=false batch_size=256 data_aug=default num_train_frames=1000000 experiment=$tag seed=$seed replay_buffer_num_workers=0
