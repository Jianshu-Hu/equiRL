#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_dmc/drqv2
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=reacher_hard_inv_equi_drq_with_ssl_convtrans_hidden_64_batch_128
seed=1

echo "start running $tag with seed $seed"
python train.py task=reacher_hard with_decoder=true decoder_type=2 batch_size=128 agent._target_=in_eq_drqv2.InvEquiDrQV2Agent num_train_frames=1000000 experiment=$tag seed=$seed replay_buffer_num_workers=0
