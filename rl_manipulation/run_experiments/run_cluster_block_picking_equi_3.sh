#!/bin/bash

cd /bigdata/users/jhu/equiRL/rl_manipulation/scripts
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate equiRL

tag=block_picking_equi_d4
seed=3

echo "start running $tag with seed $seed"
python main.py --env=close_loop_block_picking --num_objects=1 --alg=sacfd --model=equi_both_d --max_train_step=5000 --planner_episode=50 --device=cuda --view_type=camera_side_rgbd --aug=t --seed=$seed --log_pre=../logs/
