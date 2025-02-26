import os
import sys
import time
import copy
from tqdm import tqdm

sys.path.append('/')
sys.path.append('../..')
sys.path.append('..')
from rl_manipulation.utils.parameters import *
from rl_manipulation.storage.buffer import QLearningBufferExpert, QLearningBuffer
from rl_manipulation.storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
from rl_manipulation.storage.aug_buffer import QLearningBufferAug
from rl_manipulation.storage.per_aug_buffer import PrioritizedQLearningBufferAug
from rl_manipulation.utils.logger import Logger
from rl_manipulation.utils.schedules import LinearSchedule
from rl_manipulation.utils.env_wrapper import EnvWrapper

from rl_manipulation.utils.create_agent import createAgent
import threading

from rl_manipulation.utils.torch_utils import ExpertTransition, augmentBuffer

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(agent, replay_buffer, logger, p_beta_schedule):
    if buffer_type[:3] == 'per':
        beta = p_beta_schedule.value(logger.num_training_steps)
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
        loss, td_error = agent.update(batch)
        new_priorities = np.abs(td_error.cpu()) + np.stack([t.expert for t in batch]) * per_expert_eps + per_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)
        logger.expertSampleBookkeeping(
            np.stack(list(zip(*batch))[-1]).sum() / batch_size)
    else:
        batch = replay_buffer.sample(batch_size)
        loss, td_error = agent.update(batch)

    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def preTrainCURLStep(agent, replay_buffer, logger):
    if buffer_type[:3] == 'per':
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, per_beta)
    else:
        batch = replay_buffer.sample(batch_size)
    loss = agent.updateCURLOnly(batch)
    logger.trainingBookkeeping(loss, 0)

def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLearningCurve(20)
    logger.saveLossCurve(100)
    logger.saveTdErrorCurve(100)
    logger.saveStepLeftCurve(100)
    logger.saveExpertSampleCurve(100)
    logger.saveEvalCurve()
    logger.saveRewards()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveEvalRewards()

def evaluate(envs, agent, logger):
    states, obs = envs.reset()
    evaled = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    eval_rewards = []
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    while evaled < num_eval_episodes:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + gamma * R
                eval_rewards.append(R)
                temp_reward[i] = []
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)
    logger.eval_rewards.append(np.mean(eval_rewards[:num_eval_episodes]))
    if not no_bar:
        eval_bar.close()

def countParameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    print('creating envs')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    eval_envs = EnvWrapper(num_eval_processes, simulator, env, env_config, planner_config)

    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)
    # .train() is required for equivariant network
    agent.train()
    eval_agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)
    if load_t_r_model_pre:
        agent.loadTransAndRewardModel(load_t_r_model_pre)

    # logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join(log_pre, '{}_{}'.format(alg, model))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_train_step, gamma, seed, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'per':
        replay_buffer = PrioritizedQLearningBuffer(buffer_size, per_alpha, NORMAL)
    elif buffer_type == 'per_expert':
        replay_buffer = PrioritizedQLearningBuffer(buffer_size, per_alpha, EXPERT)
    elif buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    elif buffer_type == 'normal':
        replay_buffer = QLearningBuffer(buffer_size)
    elif buffer_type == 'aug':
        replay_buffer = QLearningBufferAug(buffer_size, aug_n=buffer_aug_n)
    elif buffer_type == 'per_expert_aug':
        replay_buffer = PrioritizedQLearningBufferAug(buffer_size, per_alpha, EXPERT, aug_n=buffer_aug_n)
    else:
        raise NotImplementedError
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)
    p_beta_schedule = LinearSchedule(schedule_timesteps=max_train_step, initial_p=per_beta, final_p=1.0)

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    if load_buffer is not None and not load_sub:
        logger.loadBuffer(replay_buffer, load_buffer, load_n)

    if planner_episode > 0 and not load_sub:
        planner_envs = envs
        planner_num_process = num_processes
        j = 0
        states, obs = planner_envs.reset()
        s = 0
        if not no_bar:
            planner_bar = tqdm(total=planner_episode)
        local_transitions = [[] for _ in range(planner_num_process)]
        while j < planner_episode:
            plan_actions = planner_envs.getNextAction()
            planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
            states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
            for i in range(planner_num_process):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), planner_actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                              np.array(100), np.array(1))
                # if obs_type == 'pixel':
                    # transition = normalizeTransition(transition)
                # replay_buffer.add(transition)
                local_transitions[i].append(transition)
            states = copy.copy(states_)
            obs = copy.copy(obs_)

            for i in range(planner_num_process):
                if dones[i] and rewards[i]:
                    for t in local_transitions[i]:
                        replay_buffer.add(t)
                    local_transitions[i] = []
                    j += 1
                    s += 1
                    if not no_bar:
                        planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                        planner_bar.update(1)
                    if j == planner_episode:
                        break
                elif dones[i]:
                    local_transitions[i] = []
        if not no_bar:
            planner_bar.close()

        if expert_aug_n > 0:
            augmentBuffer(replay_buffer, buffer_aug_type, expert_aug_n)

        if alg in ['curl_sac', 'curl_sacfd', 'curl_sacfd_mean']:
            if not no_bar:
                pre_train_bar = tqdm(total=1600)
            for _ in range(1600):
                preTrainCURLStep(agent, replay_buffer, logger)
                if not no_bar:
                    pre_train_bar.update(1)

    # pre train
    if pre_train_step > 0 and not load_sub and not load_model_pre:
        pbar = tqdm(total=pre_train_step)
        for i in range(pre_train_step):
            t0 = time.time()
            train_step(agent, replay_buffer, logger, p_beta_schedule)
            if logger.num_training_steps % 1000 == 0:
                logger.saveLossCurve(100)
                logger.saveTdErrorCurve(100)
            if not no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update()

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(args, envs, agent, replay_buffer)
                exit(0)
        pbar.close()
        logger.saveModel(0, 'pretrain', agent)

    # pre train enc when transferring
    if load_t_r_model_pre and pre_train_enc_step > 0:
        if not no_bar:
            pbar = tqdm(total=pre_train_enc_step)
        for i in range(pre_train_enc_step):
            if buffer_type[:3] == 'per':
                batch, weights, batch_idxes = replay_buffer.sample(batch_size, per_beta)
            else:
                batch = replay_buffer.sample(batch_size)
            loss = agent.updateModel(batch)
            logger.trainingBookkeeping(loss, 0)
            if not no_bar:
                pbar.update(1)
        if not no_bar:
            pbar.close()

    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    states, obs = envs.reset()
    while logger.num_training_steps <= max_train_step:
        # perform model training
        if alg.find('reg') > -1 and logger.num_training_steps > 0 and train_model_freq > 0 and \
                logger.num_training_steps % train_model_freq == 0 and \
                logger.model_train_iter < logger.num_training_steps / train_model_freq:
            # initiate model learning if there is at least two third of the time remaining on cluster. ow train on next job
            if (time.time() - start_time)/3600 < time_limit/3:
                logger.model_train_iter += 1
                agent.trainModel(logger, replay_buffer.sample(len(replay_buffer)), 256, max_epochs=train_model_max_epoch)
            else:
                break

        if logger.num_training_steps == max_train_step:
            break

        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_training_steps)

        is_expert = 0

        # simulate actions
        if simulate_n > 0 and envs.canSimulate():
            sim_obs = obs
            sim_states = states
            for _ in range(simulate_n):
                if not envs.canSimulate():
                    envs.resetSimPose()
                    sim_obs = obs
                    sim_states = states
                sim_actions_star_idx, sim_actions_star = agent.getEGreedyActions(sim_states, sim_obs, eps)
                sim_states_, sim_obs_, sim_rewards, sim_dones = envs.simulate(sim_actions_star)

                if not alg[:2] == 'bc':
                    for i in range(num_processes):
                        transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), sim_actions_star_idx[i].numpy(),
                                                      sim_rewards[i].numpy(), sim_states_[i].numpy(), sim_obs_[i].numpy(), sim_dones[i].numpy(),
                                                      np.array(100), np.array(is_expert))
                        # if obs_type == 'pixel':
                        #     transition = normalizeTransition(transition)
                        replay_buffer.add(transition)
                        # insert extra training steps after simulation
                        if train_simulate:
                            if len(replay_buffer) >= training_offset:
                                for training_iter in range(training_iters):
                                    train_step(agent, replay_buffer, logger, p_beta_schedule)

                sim_obs = sim_obs_
                sim_states = sim_states_

        actions_star_idx, actions_star = agent.getEGreedyActions(states, obs, eps)

        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger, p_beta_schedule)

        states_, obs_, rewards, dones = envs.stepWait()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                obs_[idx] = reset_obs_[j]

        if not alg[:2] == 'bc':
            for i in range(num_processes):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                              np.array(100), np.array(is_expert))
                # if obs_type == 'pixel':
                #     transition = normalizeTransition(transition)
                replay_buffer.add(transition)
        logger.stepBookkeeping(rewards.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Action Step:{}; Episode: {}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.num_episodes, logger.getCurrentAvgReward(100), logger.eval_rewards[-1] if len(logger.eval_rewards) > 0 else 0, eps, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps-pbar.n)
        logger.num_steps += num_processes

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent, logger))
            eval_thread.start()
            # evaluate(eval_envs, agent, logger)

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()
    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)
    if logger.num_training_steps >= max_train_step:
        logger.saveResult()
    envs.close()
    eval_envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()