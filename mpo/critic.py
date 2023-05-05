import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal, Categorical


class Critic(nn.Module):
    def __init__(self, device, observations, actions):
        super(Critic, self).__init__()
        self.device = device
        self.observations = observations
        self.actions = actions
        # https://arxiv.org/pdf/2304.13653.pdf, page 34
        self.input = torch.nn.LayerNorm(observations + actions)
        self.layer_1 = nn.Linear(observations + actions, 400)
        self.layer_2 = nn.Linear(400, 400)
        self.layer_3 = nn.Linear(400, 300)
        self.layer_4 = nn.Linear(300, 1)

        # Initialize weights
        torch.nn.init.uniform_(self.layer_1.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_2.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_3.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_4.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.tanh(self.input(x))
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x


class CriticOptimizerDiscrete:
    def __init__(self, critic, target_critic, target_actor, discount_factor, lr):
        self.critic = critic
        self.discount_factor = discount_factor
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.norm_loss_q = nn.SmoothL1Loss()
        self.action_eye = torch.eye(critic.actions)

    def train(self, batch_state, batch_action, batch_reward, batch_next_state, stats):
        # Convert to tensors
        batch_state = torch.from_numpy(np.stack(batch_state)) \
            .type(torch.float32) \
            .to(self.critic.device)
        batch_action = torch.from_numpy(np.stack(batch_action)) \
            .type(torch.float32) \
            .to(self.critic.device)
        batch_next_state = torch.from_numpy(np.stack(batch_next_state)) \
            .type(torch.float32) \
            .to(self.critic.device)
        batch_reward = torch.from_numpy(np.stack(batch_reward)) \
            .type(torch.float32) \
            .to(self.critic.device)

        # Collect shapes
        batch_size = batch_state.size(0)
        observations = self.critic.observations
        actions = self.critic.actions

        # Compute TD
        with torch.no_grad():
            pi_prob = self.target_actor.forward(batch_next_state)  # (B, da)
            pi = Categorical(probs=pi_prob)  # (B,)
            pi_log_prob = pi.expand((actions, batch_size)).log_prob(
                torch.arange(actions)[..., None].expand(-1, batch_size)  # (da, B)
            ).exp().transpose(0, 1)  # (B, da)
            sampled_next_actions = self.action_eye[None, ...].expand(batch_size, -1, -1)  # (B, da, da)
            expanded_next_states = batch_next_state[:, None, :].expand(-1, actions, -1)  # (B, da, ds)
            expected_next_q = (
                    self.target_critic.forward(
                        expanded_next_states.reshape(-1, observations),  # (B * da, ds)
                        sampled_next_actions.reshape(-1, actions)  # (B * da, da)
                    ).reshape(batch_size, actions) * pi_log_prob  # (B, da)
            ).sum(dim=-1)  # (B,)
            y = batch_reward + self.discount_factor * expected_next_q

        # Optimize critic
        self.critic_optimizer.zero_grad()
        t = self.critic(batch_state, self.action_eye[batch_action.long()]).squeeze(-1)  # (B,)
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()

        # Return loss
        return loss, y


class CriticOptimizerContinuous:
    def __init__(self, critic, target_critic, target_actor, discount_factor, sample_num, lr):
        self.critic = critic
        self.discount_factor = discount_factor
        self.sample_num = sample_num
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.norm_loss_q = nn.SmoothL1Loss()

    def train(self, batch_state, batch_action, batch_reward, batch_next_state, stats):
        # Convert to tensors
        batch_state = torch.from_numpy(np.stack(batch_state)) \
            .type(torch.float32) \
            .to(self.critic.device)
        batch_action = torch.from_numpy(np.stack(batch_action)) \
            .type(torch.float32) \
            .to(self.critic.device)
        batch_next_state = torch.from_numpy(np.stack(batch_next_state)) \
            .type(torch.float32) \
            .to(self.critic.device)
        batch_reward = torch.from_numpy(np.stack(batch_reward)) \
            .type(torch.float32) \
            .to(self.critic.device)

        # Collect shapes
        batch_size = batch_state.size(0)
        observations = self.critic.observations
        actions = self.critic.actions

        # Compute TD
        with torch.no_grad():
            pi_mu, pi_a = self.target_actor.forward(batch_next_state)  # (B,)
            pi = MultivariateNormal(pi_mu, scale_tril=pi_a)  # (B,)
            sampled_next_actions = pi.sample((self.sample_num,)).transpose(0, 1)  # (B, sample_num, da)
            expanded_next_states = batch_next_state[:, None, :].expand(-1, self.sample_num, -1)  # (B, sample_num, ds)
            expected_next_q = self.target_critic.forward(
                expanded_next_states.reshape(-1, observations),  # (B * sample_num, ds)
                sampled_next_actions.reshape(-1, actions)  # (B * sample_num, da)
            ).reshape(batch_size, self.sample_num).mean(dim=1)  # (B,)
            y = batch_reward + self.discount_factor * expected_next_q

        # Optimize critic
        self.critic_optimizer.zero_grad()
        t = self.critic(batch_state, batch_action).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()

        # Return loss
        return loss, y
