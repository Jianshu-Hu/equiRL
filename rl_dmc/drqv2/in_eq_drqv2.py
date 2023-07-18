import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import utils
import kornia
import numpy as np

from e2cnn import gspaces
from e2cnn import nn as e2nn



class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class InvEquiEncoder(torch.nn.Module):
    def __init__(self, obs_shape, out_dim, hidden_dim=32):
        super().__init__()
        self.obs_shape = obs_shape
        self.out_dim = out_dim
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)
        in_type = e2nn.FieldType(self.r2_act, obs_shape[0]*[self.r2_act.trivial_repr])
        self.input_type = in_type

        self.repr_dim = out_dim+2

        self.eps = 1e-5

        # convolution 1
        out_scalar_fields = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = self.get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = self.get_non_linearity(out_scalar_fields, out_vector_field)

        self.block1 = e2nn.SequentialModule(
            #nn.MaskModule(in_type, 29, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 2
        in_type = out_type
        out_scalar_fields = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = self.get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = self.get_non_linearity(out_scalar_fields, out_vector_field)

        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool1 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        in_type = out_type
        out_scalar_fields = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = self.get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = self.get_non_linearity(out_scalar_fields, out_vector_field)

        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 4
        in_type = out_type
        out_scalar_fields = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = self.get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = self.get_non_linearity(out_scalar_fields, out_vector_field)

        self.block4 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool2 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        in_type = out_type
        out_scalar_fields = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = self.get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = self.get_non_linearity(out_scalar_fields, out_vector_field)

        self.block5 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 6
        in_type = out_type
        out_scalar_fields = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = self.get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = self.get_non_linearity(out_scalar_fields, out_vector_field)

        self.block6 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # convolution 7 --> out
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = e2nn.FieldType(self.r2_act, out_dim * [self.r2_act.trivial_repr])
        out_vector_field = e2nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field

        self.block7 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
        )

    def get_non_linearity(self, scalar_fields, vector_fields):
        out_type = scalar_fields + vector_fields
        relu = e2nn.ReLU(scalar_fields)
        norm_relu = e2nn.NormNonLinearity(vector_fields)
        nonlinearity = e2nn.MultipleModule(
            out_type,
            ['relu'] * len(scalar_fields) + ['norm'] * len(vector_fields),
            [(relu, 'relu'), (norm_relu, 'norm')]
        )
        return nonlinearity

    def get_batch_norm(self, scalar_fields, vector_fields):
        out_type = scalar_fields + vector_fields
        batch_norm = e2nn.InnerBatchNorm(scalar_fields)
        norm_batch_norm = e2nn.NormBatchNorm(vector_fields)
        batch_norm = e2nn.MultipleModule(
            out_type,
            ['bn'] * len(scalar_fields) + ['nbn'] * len(vector_fields),
            [(batch_norm, 'bn'), (norm_batch_norm, 'nbn')]
        )
        return batch_norm

    def forward(self, x: torch.Tensor):
        x = x / 255.0 - 0.5
        x = e2nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)

        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        # normalize the vector corresponding to rotation
        x_1 = x_1 / (torch.norm(x_1, dim=-1, keepdim=True) + self.eps)
        x = torch.cat([x_0, x_1], dim=-1)

        return x


class InvEquiDecoder1(torch.nn.Module):
    def __init__(self, obs_shape, input_size, hidden_size=64):
        super().__init__()
        self.obs_shape = obs_shape
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0,),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, obs_shape[0], kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor, rot_img=True):
        x = x.unsqueeze(-1).unsqueeze(-1)  # [bz, emb_dim, 1, 1]
        x = x.expand(-1, -1, 3, 3)

        x = self.block1(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block2(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block3(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block4(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block5(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block6(x)
        init_index = (96-self.obs_shape[-1])//2
        x = x[:, :, init_index:init_index+self.obs_shape[-1], init_index:init_index+self.obs_shape[-1]]
        x = torch.sigmoid(x)

        if rot_img:
            rot = get_rotation_matrix(v)
            x = rot_img(x, rot)
        return x


class InvEquiDecoder2(torch.nn.Module):
    def __init__(self, obs_shape, input_size, hidden_size=64):
        super().__init__()
        self.obs_shape = obs_shape
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size * 4 * 4),
            torch.nn.ReLU()
        )
        self.hidden_size = hidden_size

        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=1, stride=2, padding=0),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, obs_shape[0], kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor, rot_img=True):
        x = self.linear(x)  # [bz, hidden_size*4*4]
        x = x.view(-1, self.hidden_size, 4, 4)  # [bz, hidden_size, 4, 4]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        init_index = (97-self.obs_shape[-1])//2
        x = x[:, :, init_index:init_index+self.obs_shape[-1], init_index:init_index+self.obs_shape[-1]]
        x = torch.sigmoid(x)

        if rot_img:
            rot = get_rotation_matrix(v)
            x = rot_img(x, rot)
        return x


def get_rotation_matrix(v):
    rot = torch.stack((
        torch.stack((v[:, 0], v[:, 1]), dim=-1),
        torch.stack((-v[:, 1], v[:, 0]), dim=-1),
        torch.zeros(v.size(0), 2).type_as(v)
    ), dim=-1)
    return rot


def rot_img(x, rot):
    grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class InvEquiDrQV2Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        encoder_hidden_dim,
        encoder_out_dim,
        mixed_precision,
        task_name,
        aug_K,
        with_decoder,
        decoder_type,
        ssl
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = InvEquiEncoder(obs_shape=obs_shape, out_dim=encoder_out_dim, hidden_dim=encoder_hidden_dim).to(
            device
        )
        if with_decoder:
            if decoder_type == 1:
                self.decoder = InvEquiDecoder1(obs_shape=obs_shape, input_size=encoder_out_dim).to(device)
            elif decoder_type == 2:
                self.decoder = InvEquiDecoder2(obs_shape=obs_shape, input_size=encoder_out_dim).to(device)

        self.actor = Actor(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        if with_decoder:
            self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # scaler for mixed precision
        self.scaler = GradScaler()
        self.mixed_precision = mixed_precision

        # data augmentation
        self.task_name = task_name
        self.aug_K = aug_K
        self.shift_aug = RandomShiftsAug(pad=4)
        self.rot_aug = kornia.augmentation.RandomRotation(degrees=180)

        # self supervised loss
        self.with_decoder = with_decoder
        self.ssl = ssl

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        if self.with_decoder:
            self.decoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.training = False
        self.encoder.eval()
        if self.with_decoder:
            self.decoder.eval()
        self.actor.eval()
        self.critic.eval()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        target_all = []
        with torch.no_grad():
            for k in range(self.aug_K):
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs[k], stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Q1, target_Q2 = self.critic_target(next_obs[k], next_action)
                target_V = torch.min(target_Q1, target_Q2)
                target_Q = reward + (discount * target_V)
                target_all.append(target_Q)
            avg_target_Q = sum(target_all)/self.aug_K

        # with autocast(enabled=self.mixed_precision):
        critic_loss_all = []
        for k in range(self.aug_K):
            Q1, Q2 = self.critic(obs[k], action[k])
            critic_loss = F.mse_loss(Q1, avg_target_Q) + F.mse_loss(Q2, avg_target_Q)
            critic_loss_all.append(critic_loss)
        avg_critic_loss = sum(critic_loss_all)/self.aug_K

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = avg_critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        avg_critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # self.scaler.scale(critic_loss).backward()
        # self.scaler.step(self.critic_opt)
        # self.scaler.step(self.encoder_opt)

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # with autocast(enabled=self.mixed_precision):
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)

        actor_loss.backward()
        self.actor_opt.step()

        # self.scaler.scale(actor_loss).backward()
        # self.scaler.step(self.actor_opt)

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_encoder_decoder(self, obs):
        metrics = dict()

        feature = self.encoder(obs)
        emb, v = feature[:, :self.encoder.out_dim], feature[:, self.encoder.out_dim:]
        y = self.decoder(emb, v)

        loss = torch.nn.functional.mse_loss(y, obs / 255.0)

        # optimize encoder and decoder
        self.encoder_opt.zero_grad(set_to_none=True)
        self.decoder_opt.zero_grad(set_to_none=True)

        loss.backward()
        self.decoder_opt.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics["encoder_decoder_loss"] = loss.item()

        return metrics

    def update_encoder_ssl(self, obs):
        metrics = dict()

        feature = self.encoder(obs)
        emb, v = feature[:, :self.encoder.out_dim], feature[:, self.encoder.out_dim:]

        theta = torch.tensor(np.random.rand(obs.size(0)) * 360.0).float().to(self.device)
        delta_v = torch.concat((torch.cos(theta).unsqueeze(1), torch.sin(theta).unsqueeze(1)), 1)
        rot_matrix = torch.stack((
            torch.stack((delta_v[:, 0], delta_v[:, 1]), dim=-1),
            torch.stack((-delta_v[:, 1], delta_v[:, 0]), dim=-1)
        ), dim=-1)

        rot_v = torch.matmul(rot_matrix, v.unsqueeze(-1)).squeeze(-1)
        rot_feature = torch.concat((emb, rot_v), dim=-1)

        # target feature
        with torch.no_grad():
            rot = get_rotation_matrix(delta_v)
            rot_obs = rot_img(obs, rot)
            target_feature = self.encoder(rot_obs)

        loss = torch.nn.functional.mse_loss(rot_feature, target_feature)

        # optimize encoder
        self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_opt.step()

        if self.use_tb:
            metrics["encoder_ssl"] = loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment the obs, next obs for training the actor and critic
        obs_all = []
        action_all = []
        next_obs_all = []
        for k in range(self.aug_K):
            obs_aug = self.shift_aug(obs.clone().float())
            next_obs_aug = self.shift_aug(next_obs.clone().float())
            # encoder
            obs_all.append(self.encoder(obs_aug))
            with torch.no_grad():
                action_all.append(action)
                next_obs_all.append(self.encoder(next_obs_aug))

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs_all, action_all, reward, discount, next_obs_all, step)
        )

        # update actor
        metrics.update(self.update_actor(obs_all[0].detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        if self.with_decoder:
            # augment the obs for training the encoder, decoder
            obs_rot_aug = self.rot_aug(obs.clone().float())
            metrics.update(self.update_encoder_decoder(obs_rot_aug))

        if self.ssl == 1:
            metrics.update(self.update_encoder_ssl(obs.float()))

        # self.scaler.update()

        return metrics

    def save(self):
        save_dict = dict()
        self.eval()
        self.critic_target.eval()

        # model
        save_dict["agent.encoder"] = self.encoder.state_dict()
        if self.with_decoder:
            save_dict["agent.decoder"] = self.decoder.state_dict()
        save_dict["agent.actor"] = self.actor.state_dict()
        save_dict["agent.critic"] = self.critic.state_dict()
        save_dict["agent.critic_target"] = self.critic_target.state_dict()

        # optimizers
        save_dict["agent.encoder_opt"] = self.encoder_opt.state_dict()
        if self.with_decoder:
            save_dict["agent.decoder_opt"] = self.decoder_opt.state_dict()
        save_dict["agent.actor_opt"] = self.actor_opt.state_dict()
        save_dict["agent.critic_opt"] = self.critic_opt.state_dict()

        self.train()
        self.critic_target.train()

        return save_dict

    def load(self, state_dict):
        self.eval()
        self.critic_target.eval()

        # model
        self.encoder.load_state_dict(state_dict["agent.encoder"])
        if self.with_decoder:
            self.decoder.load_state_dict(state_dict["agent.decoder"])
        self.actor.load_state_dict(state_dict["agent.actor"])
        self.critic.load_state_dict(state_dict["agent.critic"])
        self.critic_target.load_state_dict(state_dict["agent.critic_target"])

        # optimizers
        self.encoder_opt.load_state_dict(state_dict["agent.encoder_opt"])
        if self.with_decoder:
            self.decoder_opt.load_state_dict(state_dict["agent.decoder_opt"])
        self.actor_opt.load_state_dict(state_dict["agent.actor_opt"])
        self.critic_opt.load_state_dict(state_dict["agent.critic_opt"])

        self.encoder.to(self.device)
        if self.with_decoder:
            self.decoder.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.train()
        self.critic_target.train()
