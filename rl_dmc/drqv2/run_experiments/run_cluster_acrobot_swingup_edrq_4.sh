#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=acrobot_swingup_flip_edrq_shift
seed=4

echo "start running $tag with seed $seed"
python train.py task=acrobot_swingup num_train_frames=1100000 agent._target_=eq_drqv2.EquiDrQV2Agent agent.encoder_hidden_dim=22 agent.encoder_out_dim=22 agent.hidden_dim=720 experiment=$tag seed=$seed replay_buffer_num_workers=0
