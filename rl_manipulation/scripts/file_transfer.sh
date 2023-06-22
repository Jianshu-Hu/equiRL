#source=/bigdata/users/jhu/dmcontrol-generalization/outputs/
#target=~/PycharmProjects/dmcontrol-generalization/outputs/

source2=/bigdata/users/jhu/equiRL/rl_manipulation/logs/saved_logs/
target2=~/PycharmProjects/equiRL/rl_manipulation/saved_logs/

file_name=*
file_name2=*.png

#scp -r jhu@aaal.ji.sjtu.edu.cn:$source$file_name $target
scp -r jhu@aaal.ji.sjtu.edu.cn:$source2$file_name2 $target2
