#source=/bigdata/users/jhu/dmcontrol-generalization/outputs/
#target=~/PycharmProjects/dmcontrol-generalization/outputs/

source2=/bigdata/users/jhu/equiRL/rl_dmc/drqv2/saved_figs/
target2=~/PycharmProjects/equiRL/rl_dmc/drqv2/saved_figs/

file_name=*
file_name2=*

scp -r jhu@aaal.ji.sjtu.edu.cn:$source$file_name $target
scp -r jhu@aaal.ji.sjtu.edu.cn:$source2$file_name2 $target2