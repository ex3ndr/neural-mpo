import torch
import numpy as np
from torch.distributions import Categorical, MultivariateNormal

from mpo.math.dual import solve_dual_discrete, solve_dual_continuous


class EStepDiscrete:

    def __init__(self, observations, actions, target_actor, target_critic, dual_constraint):
        self.observations = observations
        self.actions = actions
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.eta = np.random.rand()
        self.A_eye = torch.eye(actions)
        self.dual_constraint = dual_constraint

    def train(self, batch_state, stats):
        batch_size = len(batch_state)
        actions = torch.arange(self.actions)[..., None].expand(self.actions, batch_size)

        # Collect actions
        b_p = self.target_actor.forward(batch_state)  # (K, da)
        b = Categorical(probs=b_p)  # (K,)
        b_prob = b.expand((self.actions, batch_size)).log_prob(actions).exp()  # (da, K)

        # Collect next states
        expanded_actions = self.A_eye[None, ...] \
            .expand(batch_size, -1, -1)  # (K, da, da)
        expanded_states = batch_state \
            .reshape(batch_size, 1, self.observations) \
            .expand((batch_size, self.actions, self.observations))  # (K, da, ds)
        target_q = (
            self.target_critic.forward(
                expanded_states.reshape(-1, self.observations),  # (K * da, ds)
                expanded_actions.reshape(-1, self.actions)  # (K * da, da)
            ).reshape(batch_size, self.actions)  # (K, da)
        ).transpose(0, 1)  # (da, K)

        # Solve eta
        self.eta = solve_dual_discrete(target_q, b_prob, self.eta, self.dual_constraint)

        # Calculate qij
        qij = torch.softmax(target_q / self.eta, dim=0)  # (N, K) or (da, K)

        return qij, actions, b_p


class EStepContinuous:
    def __init__(self, observations, actions, target_actor, target_critic, dual_constraint, action_samples):
        self.observations = observations
        self.actions = actions
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.action_samples = action_samples
        self.eta = np.random.rand()
        self.dual_constraint = dual_constraint

    def train(self, batch_state, stats):
        # sample N actions per state
        b_μ, b_A = self.target_actor.forward(batch_state)  # (K,)
        b = MultivariateNormal(b_μ, scale_tril=b_A)  # (K,)
        sampled_actions = b.sample((self.action_samples,))  # (N, K, da)
        expanded_states = batch_state[None, ...].expand(self.action_samples, -1, -1)  # (N, K, ds)
        target_q = self.target_critic.forward(
            expanded_states.reshape(-1, self.observations),  # (N * K, ds)
            sampled_actions.reshape(-1, self.actions)  # (N * K, da)
        ).reshape(self.action_samples, -1)  # (N, K)

        # Solve eta
        self.eta = solve_dual_continuous(target_q, self.eta, self.dual_constraint)

        # Calculate qij
        qij = torch.softmax(target_q / self.eta, dim=0)  # (N, K) or (da, K)

        return qij, sampled_actions, (b_μ, b_A)
