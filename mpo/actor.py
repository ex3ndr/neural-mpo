import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from mpo.math.cholesky import cholesky_vector_size, cholesky_vector_to_matrix_t


class ActorContinuous(nn.Module):
    def __init__(self, device, observations, actions, actions_space):
        super(ActorContinuous, self).__init__()
        self.actions = actions
        self.actions_space = actions_space
        self.observations = observations
        self.device = device

        # https://arxiv.org/pdf/2304.13653.pdf, page 34
        self.layer_1 = nn.Linear(observations, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.output_mean = nn.Linear(128, actions)
        self.output_cholesky = nn.Linear(128, cholesky_vector_size(actions))

        # Initialize weights
        torch.nn.init.uniform_(self.layer_1.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_2.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_3.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.output_mean.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.output_cholesky.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        # Network
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        mean = torch.sigmoid(self.output_mean(x))
        cholesky_vector = F.softplus(self.output_cholesky(x))

        # Vector to matrix
        cholesky = cholesky_vector_to_matrix_t(cholesky_vector, self.actions)

        return mean, cholesky

    def action_sample(self, state):
        with torch.no_grad():
            state = torch.from_numpy(np.array([state])).type(torch.float32).to(self.device)
            mean, cholesky = self.forward(state)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()

            # Re-scale
            action = action.cpu()
            action_low = torch.from_numpy(np.array([self.env.action_space.low]))
            action_high = torch.from_numpy(np.array([self.env.action_space.high]))
            action = action_low + (action_high - action_low) * action

        return action[0].numpy()

    def action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(np.array([state])).type(torch.float32).to(self.device)
            action, _ = self.forward(state)

            # Re-scale
            action = action.cpu()
            action_low = torch.from_numpy(np.array([self.env.action_space.low]))
            action_high = torch.from_numpy(np.array([self.env.action_space.high]))
            action = action_low + (action_high - action_low) * action

        return action[0].numpy()


class ActorDiscrete(nn.Module):
    def __init__(self, device, observations, actions):
        super(ActorDiscrete, self).__init__()
        self.device = device
        self.actions = actions
        self.observations = observations

        # https://arxiv.org/pdf/2304.13653.pdf, page 34
        self.layer_1 = nn.Linear(observations, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, actions)

        # Initialize weights
        torch.nn.init.uniform_(self.layer_1.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_2.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.layer_3.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.output.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x

    def action_sample(self, state):
        with torch.no_grad():
            state = torch.from_numpy(np.array([state])).type(torch.float32).to(self.device)
            p = self.forward(state)
            action_distribution = Categorical(probs=p[0])
            action = action_distribution.sample()
        return action.cpu().numpy()

    def action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(np.array([state])).type(torch.float32).to(self.device)
            p = self.forward(state)
            action_distribution = Categorical(probs=p[0])
            action = action_distribution.mode
        return action.cpu().numpy()
