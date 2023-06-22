#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_manipulation/scripts
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=block_picking_drq_flip_shift
seed=5

echo "start running $tag with seed $seed"
python main.py --env=close_loop_block_picking --aug=f --aug_type=crop_flip --num_objects=1 --alg=sacfd_drq --model=cnn_sim --max_train_step=5000 --planner_episode=50 --device=cuda --view_type=camera_side_rgbd --seed=$seed --log_pre=../logs/
