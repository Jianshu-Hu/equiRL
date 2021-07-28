from agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class SAC(A2CBase):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.policy_type == 'gaussian':
            if self.automatic_entropy_tuning is True:
                # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.target_entropy = -n_a
                self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.num_update = 0

    def initNetwork(self, actor, critic, initialize_target=True):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-5)
        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


    def getEGreedyActions(self, state, obs, eps):
        return self.getSACAction(state, obs, evaluate=False)

    def getGreedyActions(self, state, obs):
        return self.getSACAction(state, obs, evaluate=True)

    def getSACAction(self, state, obs, evaluate):
        with torch.no_grad():
            state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
            stacked = torch.cat([obs, state_tile], dim=1).to(self.device)

            if evaluate is False:
                action, _, _ = self.actor.sample(stacked)
            else:
                _, _, action = self.actor.sample(stacked)
            action = action.to('cpu')
            return self.decodeActions(*[action[:, i] for i in range(self.n_a)])

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        next_state_batch = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)
        state_batch = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            next_state_log_pi = next_state_log_pi.reshape(batch_size)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            qf1_next_target = qf1_next_target.reshape(batch_size)
            qf2_next_target = qf2_next_target.reshape(batch_size)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + non_final_masks * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.reshape(batch_size)
        qf2 = qf2.reshape(batch_size)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        with torch.no_grad():
            td_error = torch.abs(qf2 - next_q_value)

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()), td_error
