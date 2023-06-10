#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_flipr2_edrq_shift
seed=4

echo "start running $tag with seed $seed"
python train.py task=reacher_hard agent._target_=eq_drqv2.EquiDrQV2Agent agent.encoder_hidden_dim=16 agent.encoder_out_dim=16 agent.hidden_dim=512 experiment=$tag seed=$seed replay_buffer_num_workers=1
