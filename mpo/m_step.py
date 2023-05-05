import numpy as np
import torch
from torch.distributions import Categorical, MultivariateNormal
from torch.nn.utils import clip_grad_norm_

from mpo.math.kl import categorical_kl, gaussian_kl


class MStepDiscrete:
    def __init__(self, observations, actions, actor, alpha_scale, alpha_max, alpha_kl, lr):
        self.observations = observations
        self.actions = actions
        self.alpha = 0.0
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_scale = alpha_scale
        self.alpha_max = alpha_max
        self.alpha_kl = alpha_kl

    def train(self, batch_state, qij, actions, probs):
        pi_p = self.actor.forward(batch_state)  # (K, da)
        # First term of last eq of [2] p.5
        pi = Categorical(probs=pi_p)  # (K,)
        loss_p = torch.mean(qij * pi.log_prob(actions))

        kl = categorical_kl(p1=pi_p, p2=probs)
        if np.isnan(kl.item()):  # This should not happen
            raise RuntimeError('kl is nan')

        # Update lagrange multipliers by gradient descent
        # this equation is derived from last eq of [2] p.5,
        # just differentiate with respect to α
        # and update α so that the equation is to be minimized.
        self.alpha -= self.alpha_scale * (self.alpha_kl - kl).detach().item()
        self.alpha = np.clip(self.alpha, 0.0, self.alpha_max)

        # last eq of [2] p.5
        self.actor_optimizer.zero_grad()
        loss_l = -(loss_p + self.alpha * (self.alpha_kl - kl))
        loss_l.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()


class MStepContinuous:
    def __init__(self, observations, actions, actor, sample_actions, alpha_mu_scale, alpha_sigma_scale,
                 alpha_mu_max, alpha_sigma_max, alpha_kl_mu, alpha_kl_sigma, lr):
        self.observations = observations
        self.actions = actions
        self.actor = actor
        self.sample_actions = sample_actions
        self.alpha_mu = 0.0
        self.alpha_sigma = 0.0
        self.alpha_mu_scale = alpha_mu_scale
        self.alpha_sigma_scale = alpha_sigma_scale
        self.alpha_kl_mu = alpha_kl_mu
        self.alpha_kl_sigma = alpha_kl_sigma
        self.alpha_sigma_max = alpha_sigma_max
        self.alpha_mu_max = alpha_mu_max
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def train(self, batch_state, qij, actions, probs):
        mu, sigma = self.actor.forward(batch_state)
        # First term of last eq of [2] p.5
        # see also [2] 4.2.1 Fitting an improved Gaussian policy
        pi_1 = MultivariateNormal(loc=mu, scale_tril=probs[1])  # (K,)
        pi_2 = MultivariateNormal(loc=probs[0], scale_tril=sigma)  # (K,)
        loss_p = torch.mean(qij * (pi_1.log_prob(actions) + pi_2.log_prob(actions)))

        kl_mu, kl_sigma, sigma_i_det, sigma_det = gaussian_kl(mu_i=probs[0], mu=mu, a_i=probs[1], a=sigma)
        if np.isnan(kl_mu.item()):  # This should not happen
            raise RuntimeError('kl_μ is nan')
        if np.isnan(kl_sigma.item()):  # This should not happen
            raise RuntimeError('kl_Σ is nan')

        # Update lagrange multipliers by gradient descent
        # this equation is derived from last eq of [2] p.5,
        # just differentiate with respect to α
        # and update α so that the equation is to be minimized.
        self.alpha_mu -= self.alpha_mu_scale * (self.alpha_kl_mu - kl_mu).detach().item()
        self.alpha_sigma -= self.alpha_sigma_scale * (self.alpha_kl_sigma - kl_sigma).detach().item()
        self.alpha_mu = np.clip(0.0, self.alpha_mu, self.alpha_mu_max)
        self.alpha_sigma = np.clip(0.0, self.alpha_sigma, self.alpha_sigma_max)

        # last eq of [2] p.5
        self.actor_optimizer.zero_grad()
        loss_l = -(loss_p + self.alpha_mu * (self.alpha_kl_mu - kl_mu) + self.alpha_sigma * (
                self.alpha_kl_sigma - kl_sigma))
        loss_l.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
