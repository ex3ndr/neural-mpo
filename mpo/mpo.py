import os
from time import sleep
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from mpo.actor import ActorContinuous, ActorDiscrete
from mpo.critic import Critic, CriticOptimizerContinuous, CriticOptimizerDiscrete
from mpo.math.kl import gaussian_kl, categorical_kl
from mpo.replaybuffer import ReplayBuffer
from mpo.sampler import SamplerSimple


class MPO(object):
    """
    Maximum A Posteriori Policy Optimization (MPO)
    :param device:
    :param env: gym environment
    :param dual_constraint:
        (float) hard constraint of the dual formulation in the E-step
        correspond to [2] p.4 ε
    :param kl_mean_constraint:
        (float) hard constraint of the mean in the M-step
        correspond to [2] p.6 ε_μ for continuous action space
    :param kl_var_constraint:
        (float) hard constraint of the covariance in the M-step
        correspond to [2] p.6 ε_Σ for continuous action space
    :param kl_constraint:
        (float) hard constraint in the M-step
        correspond to [2] p.6 ε_π for discrete action space
    :param discount_factor: (float) discount factor used in Policy Evaluation
    :param alpha_scale: (float) scaling factor of the lagrangian multiplier in the M-step
    :param sample_episode_num: the number of sampled episodes
    :param sample_episode_maxstep: maximum sample steps of an episode
    :param sample_action_num:
    :param batch_size: (int) size of the sampled mini-batch
    :param episode_rerun_num:
    :param mstep_iteration_num: (int) the number of iterations of the M-step
    [1] https://arxiv.org/pdf/1806.06920.pdf
    [2] https://arxiv.org/pdf/1812.02256.pdf
    """

    def __init__(self,
                 device,
                 env,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.0001,
                 kl_constraint=0.01,
                 discount_factor=0.99,
                 alpha_mean_scale=1.0,
                 alpha_var_scale=100.0,
                 alpha_scale=10.0,
                 alpha_mean_max=0.1,
                 alpha_var_max=10.0,
                 alpha_max=1.0,
                 sample_episode_num=30,
                 sample_episode_maxstep=200,
                 sample_action_num=64,
                 batch_size=256,
                 episode_rerun_num=3,
                 mstep_iteration_num=5):
        self.device = device
        self.env = env
        if self.env.action_space.dtype == np.float32:
            self.continuous_action_space = True
        else:  # discrete action space
            self.continuous_action_space = False

        # the number of dimensions of state space
        self.ds = env.observation_space.shape[0]
        # the number of dimensions of action space
        if self.continuous_action_space:
            self.da = env.action_space.shape[0]
        else:  # discrete action space
            self.da = env.action_space.n

        self.ε_dual = dual_constraint
        self.ε_kl_μ = kl_mean_constraint
        self.ε_kl_Σ = kl_var_constraint
        self.ε_kl = kl_constraint
        self.γ = discount_factor
        self.α_μ_scale = alpha_mean_scale
        self.α_Σ_scale = alpha_var_scale
        self.α_scale = alpha_scale
        self.α_μ_max = alpha_mean_max
        self.α_Σ_max = alpha_var_max
        self.α_max = alpha_max
        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxstep = sample_episode_maxstep
        self.sample_action_num = sample_action_num
        self.batch_size = batch_size
        self.episode_rerun_num = episode_rerun_num
        self.mstep_iteration_num = mstep_iteration_num

        if not self.continuous_action_space:
            self.A_eye = torch.eye(self.da).to(self.device)

        # Networks
        self.critic = Critic(self.device, self.ds, self.da).to(self.device)
        self.target_critic = Critic(self.device, self.ds, self.da).to(self.device)
        if self.continuous_action_space:
            self.actor = ActorContinuous(self.device, self.ds, self.da).to(self.device)
            self.target_actor = ActorContinuous(self.device, self.ds, self.da).to(self.device)
        else:
            self.actor = ActorDiscrete(self.device, self.ds, self.da).to(self.device)
            self.target_actor = ActorDiscrete(self.device, self.ds, self.da).to(self.device)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        if self.continuous_action_space:
            self.critic_optimizer = CriticOptimizerContinuous(
                critic=self.critic,
                target_critic=self.target_critic,
                target_actor=self.target_actor,
                discount_factor=self.γ,
                sample_num=self.sample_action_num,
                lr=3e-4
            )
        else:
            self.critic_optimizer = CriticOptimizerDiscrete(
                critic=self.critic,
                target_critic=self.target_critic,
                target_actor=self.target_actor,
                discount_factor=self.γ,
                lr=3e-4
            )

        # Sampler
        self.sampler = SamplerSimple(env, sample_episode_maxstep)

        self.η = np.random.rand()
        self.α_μ = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.α_Σ = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.α = 0.0  # lagrangian multiplier for discrete action space in the M-step

        self.replaybuffer = ReplayBuffer()

        self.iteration = 1

    def train(self, iteration_num=1000, log_dir='log'):
        """
        :param iteration_num:
        :param log_dir:
        """

        model_save_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.iteration, iteration_num + 1):

            # Sample fresh episodes
            self.replaybuffer.clear()
            samples = self.sampler.sample(self.target_actor, self.sample_episode_num)
            self.replaybuffer.store_episodes(samples)

            buff_sz = len(self.replaybuffer)
            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            max_kl_μ = []
            max_kl_Σ = []
            max_kl = []
            mean_Σ_det = []

            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(
                            SubsetRandomSampler(range(buff_sz)), self.batch_size, drop_last=True),
                        desc='training {}/{}'.format(r + 1, self.episode_rerun_num)):
                    K = len(indices)  # the sample number of states
                    N = self.sample_action_num  # the sample number of actions per state
                    ds = self.ds  # the number of state space dimensions
                    da = self.da  # the number of action space dimensions

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)  # (K, ds)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(
                        self.device)  # (K, da) or (K,)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(
                        self.device)  # (K, ds)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)  # (K,)

                    # Policy Evaluation
                    # [2] 3 Policy Evaluation (Step 1)
                    loss_q, q = self.critic_optimizer.train(
                        batch_state=state_batch,
                        batch_action=action_batch,
                        batch_next_state=next_state_batch,
                        batch_reward=reward_batch,
                    )
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # E-Step of Policy Improvement
                    # [2] 4.1 Finding action weights (Step 2)
                    with torch.no_grad():
                        if self.continuous_action_space:
                            # sample N actions per state
                            b_μ, b_A = self.target_actor.forward(state_batch)  # (K,)
                            b = MultivariateNormal(b_μ, scale_tril=b_A)  # (K,)
                            sampled_actions = b.sample((N,))  # (N, K, da)
                            expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, ds)
                            target_q = self.target_critic.forward(
                                expanded_states.reshape(-1, ds),  # (N * K, ds)
                                sampled_actions.reshape(-1, da)  # (N * K, da)
                            ).reshape(N, K)  # (N, K)
                            target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, N)
                        else:  # discrete action spaces
                            # sample da actions per state
                            # Because of discrete action space, we can cover the all actions per state.
                            actions = torch.arange(da)[..., None].expand(da, K).to(self.device)  # (da, K)
                            b_p = self.target_actor.forward(state_batch)  # (K, da)
                            b = Categorical(probs=b_p)  # (K,)
                            b_prob = b.expand((da, K)).log_prob(actions).exp()  # (da, K)
                            expanded_actions = self.A_eye[None, ...].expand(K, -1, -1)  # (K, da, da)
                            expanded_states = state_batch.reshape(K, 1, ds).expand((K, da, ds))  # (K, da, ds)
                            target_q = (
                                self.target_critic.forward(
                                    expanded_states.reshape(-1, ds),  # (K * da, ds)
                                    expanded_actions.reshape(-1, da)  # (K * da, da)
                                ).reshape(K, da)  # (K, da)
                            ).transpose(0, 1)  # (da, K)
                            b_prob_np = b_prob.cpu().transpose(0, 1).numpy()  # (K, da)
                            target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, da)

                    # https://arxiv.org/pdf/1812.02256.pdf
                    # [2] 4.1 Finding action weights (Step 2)
                    #   Using an exponential transformation of the Q-values
                    if self.continuous_action_space:
                        def dual(η):
                            """
                            dual function of the non-parametric variational
                            Q = target_q_np  (K, N)
                            g(η) = η*ε + η*mean(log(mean(exp(Q(s, a)/η), along=a)), along=s)
                            For numerical stabilization, this can be modified to
                            Qj = max(Q(s, a), along=a)
                            g(η) = η*ε + mean(Qj, along=j) + η*mean(log(mean(exp((Q(s, a)-Qj)/η), along=a)), along=s)
                            """
                            max_q = np.max(target_q_np, 1)
                            return η * self.ε_dual + np.mean(max_q) \
                                + η * np.mean(np.log(np.mean(np.exp((target_q_np - max_q[:, None]) / η), axis=1)))
                    else:  # discrete action space
                        def dual(η):
                            """
                            dual function of the non-parametric variational
                            g(η) = η*ε + η*mean(log(sum(π(a|s)*exp(Q(s, a)/η))))
                            We have to multiply π by exp because this is expectation.
                            This equation is correspond to last equation of the [2] p.15
                            For numerical stabilization, this can be modified to
                            Qj = max(Q(s, a), along=a)
                            g(η) = η*ε + mean(Qj, along=j) + η*mean(log(sum(π(a|s)*(exp(Q(s, a)-Qj)/η))))
                            """
                            max_q = np.max(target_q_np, 1)
                            return η * self.ε_dual + np.mean(max_q) \
                                + η * np.mean(np.log(np.sum(
                                    b_prob_np * np.exp((target_q_np - max_q[:, None]) / η), axis=1)))

                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    qij = torch.softmax(target_q / self.η, dim=0)  # (N, K) or (da, K)

                    # M-Step of Policy Improvement
                    # [2] 4.2 Fitting an improved policy (Step 3)
                    for _ in range(self.mstep_iteration_num):
                        if self.continuous_action_space:
                            μ, A = self.actor.forward(state_batch)
                            # First term of last eq of [2] p.5
                            # see also [2] 4.2.1 Fitting an improved Gaussian policy
                            π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (K,)
                            π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (K,)
                            loss_p = torch.mean(
                                qij * (
                                        π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                                        + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                                )
                            )
                            mean_loss_p.append((-loss_p).item())

                            kl_μ, kl_Σ, Σi_det, Σ_det = gaussian_kl(
                                mu_i=b_μ, mu=μ,
                                a_i=b_A, a=A)
                            max_kl_μ.append(kl_μ.item())
                            max_kl_Σ.append(kl_Σ.item())
                            mean_Σ_det.append(Σ_det.item())

                            if np.isnan(kl_μ.item()):  # This should not happen
                                raise RuntimeError('kl_μ is nan')
                            if np.isnan(kl_Σ.item()):  # This should not happen
                                raise RuntimeError('kl_Σ is nan')

                            # Update lagrange multipliers by gradient descent
                            # this equation is derived from last eq of [2] p.5,
                            # just differentiate with respect to α
                            # and update α so that the equation is to be minimized.
                            self.α_μ -= self.α_μ_scale * (self.ε_kl_μ - kl_μ).detach().item()
                            self.α_Σ -= self.α_Σ_scale * (self.ε_kl_Σ - kl_Σ).detach().item()

                            self.α_μ = np.clip(0.0, self.α_μ, self.α_μ_max)
                            self.α_Σ = np.clip(0.0, self.α_Σ, self.α_Σ_max)

                            self.actor_optimizer.zero_grad()
                            # last eq of [2] p.5
                            loss_l = -(
                                    loss_p
                                    + self.α_μ * (self.ε_kl_μ - kl_μ)
                                    + self.α_Σ * (self.ε_kl_Σ - kl_Σ)
                            )
                            mean_loss_l.append(loss_l.item())
                            loss_l.backward()
                            clip_grad_norm_(self.actor.parameters(), 0.1)
                            self.actor_optimizer.step()
                        else:  # discrete action space
                            π_p = self.actor.forward(state_batch)  # (K, da)
                            # First term of last eq of [2] p.5
                            π = Categorical(probs=π_p)  # (K,)
                            loss_p = torch.mean(
                                qij * π.expand((da, K)).log_prob(actions)
                            )
                            mean_loss_p.append((-loss_p).item())

                            kl = categorical_kl(p1=π_p, p2=b_p)
                            max_kl.append(kl.item())

                            if np.isnan(kl.item()):  # This should not happen
                                raise RuntimeError('kl is nan')

                            # Update lagrange multipliers by gradient descent
                            # this equation is derived from last eq of [2] p.5,
                            # just differentiate with respect to α
                            # and update α so that the equation is to be minimized.
                            self.α -= self.α_scale * (self.ε_kl - kl).detach().item()

                            self.α = np.clip(self.α, 0.0, self.α_max)

                            self.actor_optimizer.zero_grad()
                            # last eq of [2] p.5
                            loss_l = -(loss_p + self.α * (self.ε_kl - kl))
                            mean_loss_l.append(loss_l.item())
                            loss_l.backward()
                            clip_grad_norm_(self.actor.parameters(), 0.1)
                            self.actor_optimizer.step()

            self.__update_param()

            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)
            if self.continuous_action_space:
                max_kl_μ = np.max(max_kl_μ)
                max_kl_Σ = np.max(max_kl_Σ)
                mean_Σ_det = np.mean(mean_Σ_det)
            else:  # discrete action space
                max_kl = np.max(max_kl)

            print('iteration :', it)
            print('  replay buffer :', buff_sz)
            print('  mean return :', mean_return)
            print('  mean reward :', mean_reward)
            print('  mean loss_q :', mean_loss_q)
            print('  mean loss_p :', mean_loss_p)
            print('  mean loss_l :', mean_loss_l)
            print('  mean est_q :', mean_est_q)
            print('  η :', self.η)
            if self.continuous_action_space:
                print('  max_kl_μ :', max_kl_μ)
                print('  max_kl_Σ :', max_kl_Σ)
                print('  mean_Σ_det :', mean_Σ_det)
                print('  α_μ :', self.α_μ)
                print('  α_Σ :', self.α_Σ)
            else:  # discrete action space
                print('  max_kl :', max_kl)
                print('  α :', self.α)

            writer.add_scalar('return', mean_return, it)
            writer.add_scalar('reward', mean_reward, it)
            writer.add_scalar('loss_q', mean_loss_q, it)
            writer.add_scalar('loss_p', mean_loss_p, it)
            writer.add_scalar('loss_l', mean_loss_l, it)
            writer.add_scalar('mean_q', mean_est_q, it)
            writer.add_scalar('η', self.η, it)
            if self.continuous_action_space:
                writer.add_scalar('max_kl_μ', max_kl_μ, it)
                writer.add_scalar('max_kl_Σ', max_kl_Σ, it)
                writer.add_scalar('mean_Σ_det', mean_Σ_det, it)
                writer.add_scalar('α_μ', self.α_μ, it)
                writer.add_scalar('α_Σ', self.α_Σ, it)
            else:
                writer.add_scalar('η_kl', max_kl, it)
                writer.add_scalar('α', self.α, it)
            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def __update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
