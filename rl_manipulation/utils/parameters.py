import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def noneOrStr(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
env_group = parser.add_argument_group('environment')
env_group.add_argument('--env', type=str, default='close_loop_block_stacking', help='block_picking, block_stacking, brick_stacking, '
                                                                         'brick_inserting, block_cylinder_stacking')
env_group.add_argument('--reward_type', type=str, default='sparse')
env_group.add_argument('--simulator', type=str, default='pybullet')
env_group.add_argument('--robot', type=str, default='kuka')
env_group.add_argument('--num_objects', type=int, default=3)
env_group.add_argument('--max_episode_steps', type=int, default=50)
env_group.add_argument('--fast_mode', type=strToBool, default=True)
env_group.add_argument('--action_sequence', type=str, default='pxyzr')
env_group.add_argument('--random_orientation', type=strToBool, default=True)
env_group.add_argument('--num_processes', type=int, default=5)
env_group.add_argument('--num_eval_processes', type=int, default=5)
env_group.add_argument('--render', type=strToBool, default=False)
env_group.add_argument('--workspace_size', type=float, default=0.4)
env_group.add_argument('--heightmap_size', type=int, default=152)
env_group.add_argument('--view_type', type=str, default='camera_side_rgbd')
env_group.add_argument('--obs_type', type=str, default='pixel')
env_group.add_argument('--transparent_bin', type=strToBool, default=False)
env_group.add_argument('--collision_penalty', type=strToBool, default=False)
env_group.add_argument('--fix_set', type=strToBool, default=False)
env_group.add_argument('--collision_terminate', type=strToBool, default=False)
env_group.add_argument('--white_plane', type=strToBool, default=False)
env_group.add_argument('--black_workspace', type=strToBool, default=False)
env_group.add_argument('--trans_plane', type=strToBool, default=False)
env_group.add_argument('--trans_robot', type=strToBool, default=False)

training_group = parser.add_argument_group('training')
training_group.add_argument('--alg', default='dqn')
training_group.add_argument('--model', type=str, default='resucat')
training_group.add_argument('--lr', type=float, default=1e-3)
training_group.add_argument('--actor_lr', type=float, default=None)
training_group.add_argument('--critic_lr', type=float, default=None)
training_group.add_argument('--gamma', type=float, default=0.99)
training_group.add_argument('--explore', type=int, default=0)
training_group.add_argument('--fixed_eps', action='store_true')
training_group.add_argument('--init_eps', type=float, default=0.)
training_group.add_argument('--final_eps', type=float, default=0.)
training_group.add_argument('--training_iters', type=int, default=1)
training_group.add_argument('--training_offset', type=int, default=100)
training_group.add_argument('--max_train_step', type=int, default=50000)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--target_update_freq', type=int, default=100)
training_group.add_argument('--save_freq', type=int, default=100)
training_group.add_argument('--action_selection', type=str, default='egreedy')
training_group.add_argument('--load_model_pre', type=str, default=None)
training_group.add_argument('--planner_episode', type=int, default=0)
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--perlin', type=float, default=0.0)
training_group.add_argument('--load_buffer', type=str, default=None)
training_group.add_argument('--load_n', type=int, default=1000000)
training_group.add_argument('--pre_train_step', type=int, default=0)
training_group.add_argument('--tau', type=float, default=1e-2)
training_group.add_argument('--init_temp', type=float, default=1e-2)
training_group.add_argument('--dpos', type=float, default=0.05)
training_group.add_argument('--drot_n', type=int, default=4)
training_group.add_argument('--demon_w', type=float, default=1)
training_group.add_argument('--equi_n', type=int, default=4)
training_group.add_argument('--n_hidden', type=int, default=64)
training_group.add_argument('--crop_size', type=int, default=128)
training_group.add_argument('--aug', type=strToBool, default=False)
training_group.add_argument('--buffer_aug_type', type=str, choices=['se2', 'so2', 't', 'dqn_c4', 'cn_vec', 'shift', 'crop', 'crop_x', 'crop_y'], default='so2')
training_group.add_argument('--aug_type', type=str, choices=['se2', 'so2', 't', 'dqn_c4', 'cn_vec', 'shift', 'crop', 'crop_x', 'crop_y', 'crop_flip'], default='crop')
training_group.add_argument('--buffer_aug_n', type=int, default=4)
training_group.add_argument('--expert_aug_n', type=int, default=0)
training_group.add_argument('--simulate_n', type=int, default=0)
training_group.add_argument('--train_simulate', type=strToBool, default=False)
training_group.add_argument('--load_t_r_model_pre', type=noneOrStr, default=None)
training_group.add_argument('--model_loss_w', type=float, default=0.1)
training_group.add_argument('--train_reg', type=strToBool, default=False)
training_group.add_argument('--train_model_freq', type=int, default=1000)
training_group.add_argument('--train_model_max_epoch', type=int, default=50)
training_group.add_argument('--pre_train_enc_step', type=int, default=0)
training_group.add_argument('--actor_channels', type=str, default='64')
training_group.add_argument('--critic_channels', type=str, default='64,64')
# sacfd2
training_group.add_argument('--critic_demo_loss', type=str, default='margin', choices=['margin', 'ce'])
training_group.add_argument('--critic_n_neg', type=int, default=2048)
training_group.add_argument('--critic_demo_w', type=float, default=0.1)
training_group.add_argument('--critic_margin_l', type=float, default=0.01)

eval_group = parser.add_argument_group('eval')
eval_group.add_argument('--eval_freq', default=200, type=int)
eval_group.add_argument('--num_eval_episodes', default=100, type=int)

margin_group = parser.add_argument_group('margin')
margin_group.add_argument('--margin', default='l', choices=['ce', 'bce', 'bcel', 'l', 'oril'])
margin_group.add_argument('--margin_l', type=float, default=0.1)
margin_group.add_argument('--margin_weight', type=float, default=0.1)
margin_group.add_argument('--margin_beta', type=float, default=100)

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--buffer', default='per_expert', choices=['normal', 'per', 'expert', 'per_expert', 'aug', 'per_expert_aug'])
buffer_group.add_argument('--per_eps', type=float, default=1e-6, help='Epsilon parameter for PER')
buffer_group.add_argument('--per_alpha', type=float, default=0.6, help='Alpha parameter for PER')
buffer_group.add_argument('--per_beta', type=float, default=0.4, help='Initial beta parameter for PER')
buffer_group.add_argument('--per_expert_eps', type=float, default=1)
buffer_group.add_argument('--batch_size', type=int, default=128)
buffer_group.add_argument('--buffer_size', type=int, default=100000)

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

test_group = parser.add_argument_group('test')
test_group.add_argument('--test', action='store_true')

args = parser.parse_args()
# env
random_orientation = args.random_orientation
reward_type = args.reward_type
env = args.env
simulator = args.simulator
num_objects = args.num_objects
max_episode_steps = args.max_episode_steps
fast_mode = args.fast_mode
action_sequence = args.action_sequence
num_processes = args.num_processes
num_eval_processes = args.num_eval_processes
render = args.render
robot = args.robot


workspace_size = args.workspace_size
workspace = np.asarray([[0.45-workspace_size/2, 0.45+workspace_size/2],
                        [0-workspace_size/2, 0+workspace_size/2],
                        [0.01, 0.25]])
heightmap_size = args.heightmap_size

heightmap_resolution = workspace_size/heightmap_size
action_space = [0, heightmap_size]
view_type = args.view_type
obs_type = args.obs_type
if env in ['close_loop_block_reaching', 'close_loop_block_picking', 'close_loop_household_picking']:
    obs_dim = 1 + 4 + 4
elif env in ['close_loop_block_pulling', 'close_loop_block_picking_corner', 'close_loop_drawer_opening']:
    obs_dim = 1 + 4 + 4*2
else:
    obs_dim = 1 + 4 + 4 * num_objects

transparent_bin = args.transparent_bin
collision_penalty = args.collision_penalty
fix_set = args.fix_set
collision_terminate = args.collision_terminate
white_plane = args.white_plane
black_workspace = args.black_workspace
trans_plane = args.trans_plane
trans_robot = args.trans_robot

######################################################################################
# training
alg = args.alg
model = args.model
lr = args.lr
actor_lr = args.actor_lr
critic_lr = args.critic_lr
if actor_lr is None:
    actor_lr = lr
if critic_lr is None:
    critic_lr = lr

gamma = args.gamma
explore = args.explore
fixed_eps = args.fixed_eps
init_eps = args.init_eps
final_eps = args.final_eps
training_iters = args.training_iters
training_offset = args.training_offset
max_train_step = args.max_train_step
device = torch.device(args.device_name)
target_update_freq = args.target_update_freq
save_freq = args.save_freq
action_selection = args.action_selection
planner_episode = args.planner_episode

load_model_pre = args.load_model_pre
is_test = args.test
note = args.note
seed = args.seed
perlin = args.perlin

tau = args.tau
init_temp = args.init_temp

demon_w = args.demon_w
equi_n = args.equi_n
n_hidden = args.n_hidden

crop_size = args.crop_size

aug = args.aug
aug_type = args.aug_type
buffer_aug_type = args.buffer_aug_type
buffer_aug_n = args.buffer_aug_n
expert_aug_n = args.expert_aug_n

simulate_n = args.simulate_n
train_simulate = args.train_simulate

load_t_r_model_pre = args.load_t_r_model_pre
model_loss_w = args.model_loss_w
if load_t_r_model_pre is not None:
    args.train_reg = True
train_reg = args.train_reg
train_model_freq = args.train_model_freq
train_model_max_epoch = args.train_model_max_epoch
pre_train_enc_step = args.pre_train_enc_step
actor_channels = args.actor_channels
actor_channels = [int(c) for c in actor_channels.split(',')]
critic_channels = args.critic_channels
critic_channels = [int(c) for c in critic_channels.split(',')]
# sacfd2
critic_demo_loss = args.critic_demo_loss
critic_n_neg = args.critic_n_neg
critic_demo_w = args.critic_demo_w
critic_margin_l = args.critic_margin_l

# eval
eval_freq = args.eval_freq
num_eval_episodes = args.num_eval_episodes

# pre train
load_buffer = args.load_buffer
load_n = args.load_n
pre_train_step = args.pre_train_step

# buffer
buffer_type = args.buffer
per_eps = args.per_eps
per_alpha = args.per_alpha
per_beta = args.per_beta
per_expert_eps = args.per_expert_eps
batch_size = args.batch_size
buffer_size = args.buffer_size

# margin
margin = args.margin
margin_l = args.margin_l
margin_weight = args.margin_weight
margin_beta = args.margin_beta

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

dpos = args.dpos
drot = np.pi/args.drot_n

######################################################################################
env_config = {'workspace': workspace, 'max_steps': max_episode_steps, 'obs_size': heightmap_size,
              'fast_mode': fast_mode,  'action_sequence': action_sequence, 'render': render, 'num_objects': num_objects,
              'random_orientation':random_orientation, 'reward_type': reward_type, 'robot': robot,
              'workspace_check': 'point', 'hard_reset_freq': 1000, 'physics_mode' : 'fast', 'view_type': view_type,
              'obs_type': obs_type, 'transparent_bin': transparent_bin, 'collision_penalty': collision_penalty,
              'fix_set': fix_set, 'collision_terminate': collision_terminate,
              'white_plane': white_plane, 'black_workspace': black_workspace, 'trans_plane': trans_plane,
              'trans_robot': trans_robot}
planner_config = {'random_orientation':random_orientation, 'dpos': dpos, 'drot': drot}
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))