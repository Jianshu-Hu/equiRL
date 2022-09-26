import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def getRewardsSingle(rewards, window=1000):
    moving_avg = []
    i = window
    while i-window < len(rewards):
        moving_avg.append(np.average(rewards[i-window:i]))
        i += window

    moving_avg = np.array(moving_avg)
    return moving_avg

def plotLearningCurveAvg(rewards, window=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    # avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    # std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(window, window * (avg_rewards.shape[0]+1), window)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def plotEvalCurveAvg(rewards, freq=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    # avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    # std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(freq, freq * (avg_rewards.shape[0]+1), freq)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def plotEvalCurve(base, step=50000, use_default_cm=False, freq=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'Equi SAC + RAD Crop': 'b',
            'RAD Crop': 'g',
            'DrQ Crop': 'r',
            'FERM': 'purple',
        }

    linestyle_map = {
    }
    name_map = {
        'Equi SAC + RAD Crop': 'Equi SAC + RAD',
        'RAD Crop': 'CNN SAC + RAD',
        'DrQ Crop': 'CNN SAC + DrQ',
        'FERM': 'FERM',
    }

    sequence = {
        'Equi SAC + RAD Crop': '0',
        'RAD Crop': '1',
        'DrQ Crop': '2',
        'FERM': '3',
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/eval_rewards.npy'))
                rs.append(r[:step//freq])
            except Exception as e:
                print(e)
                continue
        assert j == 3
        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('eval discounted reward')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    plt.ylim(bottom=-0.05)

    plt.tight_layout()
    # plt.savefig(os.path.join(base, 'eval.png'), bbox_inches='tight',pad_inches = 0)
    head, tail = os.path.split(base)
    plt.savefig(os.path.join(head, '{}.png'.format(tail)), bbox_inches='tight',pad_inches = 0)

def plotViewAngleCurve(base, step=5000, freq=200):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    color_map = {
        'equi': 'b',
        'cnn': 'g',
    }

    sequence = {
        'equi': '1',
        'cnn': '2',
    }

    angles = ['90', '75', '60', '45', '30', '15']

    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = [[] for _ in angles]
        for i, angle in enumerate(angles):
            for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method, angle))):
                try:
                    r = np.load(os.path.join(base, method, angle, run, 'info/eval_rewards.npy'))
                    rs[i].append(r[step//freq-1])
                except Exception as e:
                    print(e)
                    continue

        avg_rewards = np.mean(rs, axis=1)
        std_rewards = stats.sem(rs, axis=1)
        plt.fill_between(angles, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method])
        l = plt.plot(angles, avg_rewards, label=method, color=color_map[method], alpha=0.7)
        plt.legend(loc=4)

    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('view angle')
    # if base.find('bbp') > -1:
    plt.ylabel('eval return at {} steps'.format(step))
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    plt.ylim(bottom=-0.05)

    plt.tight_layout()
    # plt.savefig(os.path.join(base, 'eval.png'), bbox_inches='tight',pad_inches = 0)
    head, tail = os.path.split(base)
    plt.savefig(os.path.join(head, '{}_{}.png'.format(tail, step)), bbox_inches='tight',pad_inches = 0)

def plotStepRewardCurve(base, step=50000, use_default_cm=False, freq=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'dpos=0.05, drot=0.25pi': 'b',
            'dpos=0.05, drot=0.125pi': 'g',
            'dpos=0.03, drot=0.125pi': 'r',
            'dpos=0.1, drot=0.25pi': 'purple',

            'ban0': 'g',
            'ban2': 'r',
            'ban4': 'b',
            'ban8': 'purple',
            'ban16': 'orange',

            'C4': 'g',
            'C8': 'r',
            'D4': 'b',
            'D8': 'purple',

            '0': 'r',
            '10': 'g',
            '20': 'b',
            '40': 'purple',

            'sac+ban4': 'b',
            'sac+rot rad': 'g',
            'sac+rot rad+ban4': 'r',
            'sac+ban0': 'purple',

            'sac+aux+ban0': 'g',
            'sac+aux+ban4': 'r',
        }

    linestyle_map = {
    }
    name_map = {
        'ban0': 'buffer aug 0',
        'ban2': 'buffer aug 2',
        'ban4': 'buffer aug 4',
        'ban8': 'buffer aug 8',
        'ban16': 'buffer aug 16',

        'sac+ban4': 'SAC + buffer aug',
        'sac+rot rad': 'SAC + rot RAD',
        'sac+rot rad+ban4': 'SAC + rot RAD + buffer aug',
        'sac+ban0': 'SAC',

        'sac+aux+ban4': 'SAC + aux loss + buffer aug',
        'sac+aux+ban0': 'SAC + aux loss',

        'sac': 'SAC',
        'sacfd': 'SACfD',

        'sac+crop rad': 'SAC + crop RAD'
    }

    sequence = {
        'ban0': '0',
        'ban2': '1',
        'ban4': '2',
        'ban8': '3',
        'ban16': '4',

        'sac+ban0': '0',
        'sac+ban4': '1',
        'sac+aux+ban0': '2',
        'sac+aux+ban4': '3',
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                step_reward = np.load(os.path.join(base, method, run, 'info/step_reward.npy'))
                r = []
                for k in range(1, step+1, freq):
                    window_rewards = step_reward[(k <= step_reward[:, 0]) * (step_reward[:, 0] < k + freq)][:, 1]
                    if window_rewards.shape[0] > 0:
                        r.append(window_rewards.mean())
                    else:
                        break
                    # r.append(step_reward[(i <= step_reward[:, 0]) * (step_reward[:, 0] < i + freq)][:, 1].mean())
                rs.append(r)
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('discounted reward')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'step_reward.png'), bbox_inches='tight',pad_inches = 0)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def plotLearningCurve(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'cnn+bufferaug': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
        }

    linestyle_map = {

    }
    name_map = {
        'equi+bufferaug': 'Equivariant',
        'cnn+bufferaug': 'CNN',
        'cnn+rad': 'RAD',
        'cnn+drq': 'DrQ',
        'cnn+curl': 'FERM',
    }

    sequence = {
        'equi+equi': '0',
        'cnn+cnn': '1',
        'cnn+cnn+aug': '2',
        'equi_fcn_asr': '3',
        'tp': '4',

        'equi_fcn': '0',
        'fcn_si': '1',
        'fcn_si_aug': '2',
        'fcn': '3',

        'equi+deictic': '2',
        'cnn+deictic': '3',

        'q1_equi+q2_equi': '0',
        'q1_equi+q2_cnn': '1',
        'q1_cnn+q2_equi': '2',
        'q1_cnn+q2_cnn': '3',

        'q1_equi+q2_deictic': '0.5',
        'q1_cnn+q2_deictic': '4',

        'equi_fcn_': '1',

        '5l_equi_equi': '0',
        '5l_equi_deictic': '1',
        '5l_equi_cnn': '2',
        '5l_cnn_equi': '3',
        '5l_cnn_deictic': '4',
        '5l_cnn_cnn': '5',

    }

    # house1-4
    # plt.plot([0, 100000], [0.974, 0.974], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    # house1-5
    # plt.plot([0, 50000], [0.974, 0.974], label='expert', color='pink')
    # 0.004 pos noise
    # plt.plot([0, 50000], [0.859, 0.859], label='expert', color='pink')

    # house1-6 0.941

    # house2
    # plt.plot([0, 50000], [0.979, 0.979], label='expert', color='pink')
    # plt.axvline(x=20000, color='black', linestyle='--')

    # house3
    # plt.plot([0, 50000], [0.983, 0.983], label='expert', color='pink')
    # plt.plot([0, 50000], [0.911, 0.911], label='expert', color='pink')
    # 0.996
    # 0.911 - 0.01

    # house4
    # plt.plot([0, 50000], [0.948, 0.948], label='expert', color='pink')
    # plt.plot([0, 50000], [0.862, 0.862], label='expert', color='pink')
    # 0.875 - 0.006
    # 0.862 - 0.007 *
    # stack
    # plt.plot([0, 100000], [0.989, 0.989], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    # if base.find('bbp') > -1:
    plt.ylabel('discounted reward')

    # plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight',pad_inches = 0)

def plotSuccessRate(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'cnn+bufferaug': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
        }

    linestyle_map = {
    }
    name_map = {
    }

    sequence = {
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/success_rate.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    # if base.find('bbp') > -1:
    plt.ylabel('success rate')

    # plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'sr.png'), bbox_inches='tight',pad_inches = 0)

def showPerformance(base):
    methods = sorted(filter(lambda x: x[0] != '.', get_immediate_subdirectories(base)))
    for method in methods:
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                rs.append(r[-1000:].mean())
            except Exception as e:
                print(e)
        print('{}: {:.3f}'.format(method, np.mean(rs)))


def plotTDErrors():
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    base = '/media/dian/hdd/unet/perlin'
    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        if method[0] == '.' or method == 'DAGGER' or method == 'DQN':
            continue
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/td_errors.npy'))
                rs.append(getRewardsSingle(r[:120000], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('TD error')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.show()

def plotLoss(base, step):
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/losses.npy'))[:, 1]
                rs.append(getRewardsSingle(r[:step], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('loss')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    base = '/media/dian/hdd/mrun_results/transfer/iclr/equi_vs_cnn'
    envs = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for env in envs:
        plotEvalCurve(os.path.join(base, env), 10000, freq=200)

    base = '/media/dian/hdd/mrun_results/transfer/iclr/view_angle'
    envs = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for env in envs:
         plotViewAngleCurve(os.path.join(base, env), 10000 if env == 'bowl' else 5000)


