import matplotlib.pyplot as plt
import numpy as np
import math
import os

# eval_env_type = ['normal', 'color_hard', 'video_easy', 'video_hard']
eval_env_type = ['normal']


def average_over_several_runs(folder):
    mean_all = []
    std_all = []
    for env_type in range(len(eval_env_type)):
        data_all = []
        min_length = np.inf
        runs = os.listdir(folder)
        for i in range(len(runs)):
            data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
            evaluation_freq = data[2, -3]-data[1, -3]
            data_all.append(data[:, 2+env_type])
            if data.shape[0] < min_length:
                min_length = data.shape[0]
        average = np.zeros([len(runs), min_length])
        for i in range(len(runs)):
            average[i, :] = data_all[i][:min_length]
        mean = np.mean(average, axis=0)
        mean_all.append(mean)
        std = np.std(average, axis=0)
        std_all.append(std)

    return mean_all, std_all, evaluation_freq/1000


def plot_several_folders(prefix, folders, action_repeat, label_list=[], plot_or_save='save', title=""):
    plt.rcParams["figure.figsize"] = (9, 9)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        folder_name = 'saved_exps/'+prefix+folders[i]
        num_runs = len(os.listdir(folder_name))
        mean_all, std_all, eval_freq = average_over_several_runs(folder_name)
        for j in range(len(eval_env_type)):
            if len(eval_env_type) == 1:
                axs_plot = axs
            else:
                axs_plot = axs[int(j/2)][j-2*(int(j/2))]
            # plot variance
            axs_plot.fill_between(eval_freq*range(len(mean_all[j])),
                    mean_all[j] - std_all[j]/math.sqrt(num_runs),
                    mean_all[j] + std_all[j]/math.sqrt(num_runs), alpha=0.4)
            if len(label_list) == len(folders):
                # specify label
                axs_plot.plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=label_list[i])
            else:
                axs_plot.plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=folders[i])

            axs_plot.set_xlabel('frame steps/k')
            axs_plot.set_ylabel('episode reward')
            axs_plot.legend(fontsize=8)
            axs_plot.set_title(eval_env_type[j])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_figs/'+title)


prefix = 'reacher_hard/'
action_repeat = 2
# folders_1 = ['drq_shift', 'drq_rot', 'drq_flip', 'drq_flip_rot', 'drq_equi_flip', 'drq_equi_flip_rot', 'flipr2_edrq_shift']
# folders_2 = ['drq_without_pooling_shift', 'drq_without_pooling_flip',
#              'drq_without_pooling_rot', 'drq_without_pooling_flip_rot',
#              'flipr2_edrq_shift']
# folders_3 = ['drq_without_pooling_average_2_shift', 'drq_without_pooling_average_2_flip',
#              'drq_without_pooling_average_2_rot',
#              'drq_without_pooling_average_2_flip_rot',
#              'flipr2_edrq_shift']
# folders_4 = ['drq_without_pooling_flip', 'drq_without_pooling_average_2_flip', 'drq_without_pooling_flip_rot',
#              'drq_without_pooling_average_2_flip_rot',
#              'flipr2_edrq_shift']

# plot_several_folders(prefix, folders_1, action_repeat, title='reacher_hard_drq')
# plot_several_folders(prefix, folders_2, action_repeat, title='reacher_hard_without_pooling_drq')
# plot_several_folders(prefix, folders_3, action_repeat, title='reacher_hard_without_pooling_average_2_drq')
# plot_several_folders(prefix, folders_4, action_repeat, title='reacher_hard_average_2_for_equi_drq')

# folders_1 = ['drq_without_pooling_shift', 'drq_inv_equi_without_decoder', 'drq_inv_equi_with_decoder',
#              'drq_inv_equi_with_decoder_detach_encoder']
# plot_several_folders(prefix, folders_1, action_repeat, title='reacher_hard_inv_equi_drq')

# folders_1 = ['drq_without_pooling_shift', 'drq_D1_equi_encoder', 'drq_D1_equi_inv_encoder',
#              'drq_D1_equi_inv_norm_encoder',
#              'drq_D2_equi_encoder', 'drq_D2_equi_inv_encoder']
folders_1 = ['drq_without_pooling_shift', 'drq_D1_inv_encoder', 'drq_D1_equi_encoder', 'drq_D2_inv_encoder',
             'drq_D2_equi_encoder', 'drq_D2_inv_norm_encoder', 'drq_D4_inv_encoder', 'drq_D8_inv_encoder',
             'drq_D1_equi_C4_inv_encoder']
plot_several_folders(prefix, folders_1, action_repeat, title='reacher_hard_encoder')


prefix = 'cheetah_run/'
action_repeat = 2
# folders_1 = ['drq_shift_batch_128', 'drq_inv_equi_without_decoder', 'drq_inv_equi_with_decoder', 'drq_inv_equi_with_decoder_detach_encoder']
# plot_several_folders(prefix, folders_1, action_repeat, title='cheetah_run_inv_equi_drq')

# folders_1 = ['drq_shift', 'drq_D1_equi_encoder', 'drq_D1_equi_inv_encoder', 'drq_D1_equi_inv_norm_encoder',
#              'drq_D2_equi_encoder', 'drq_D2_equi_inv_encoder']
folders_1 = ['drq_shift', 'drq_D1_equi_encoder', 'drq_D1_inv_encoder', 'drq_D2_equi_encoder', 'drq_D2_inv_encoder']
plot_several_folders(prefix, folders_1, action_repeat, title='cheetah_run_encoder')

prefix = 'acrobot_swingup/'
action_repeat = 2
# folders_1 = ['drq_shift', 'drq_inv_equi_without_decoder']
# plot_several_folders(prefix, folders_1, action_repeat, title='acrobot_swingup_inv_equi_drq')

# folders_1 = ['drq_shift', 'drq_D1_equi_encoder', 'drq_D2_equi_encoder',
#              'drq_D1_equi_inv_encoder', 'drq_D2_equi_inv_encoder', 'drq_D1_equi_inv_norm_encoder']
folders_1 = ['drq_shift', 'drq_D1_equi_encoder', 'drq_D1_inv_encoder', 'drq_D2_equi_encoder', 'drq_D2_inv_encoder',
             'drq_D4_equi_encoder', 'drq_D2_equi_norm_encoder']
plot_several_folders(prefix, folders_1, action_repeat, title='acrobot_swingup_encoder')