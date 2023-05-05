import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from mpo.actor import ActorContinuous, ActorDiscrete
from mpo.critic import Critic, CriticOptimizerContinuous, CriticOptimizerDiscrete
from mpo.e_step import EStepContinuous, EStepDiscrete
from mpo.experience import ExperienceBuffer
from mpo.m_step import MStepDiscrete, MStepContinuous
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
        if self.continuous_action_space:
            self.critic_optimizer = CriticOptimizerContinuous(
                critic=self.critic,
                target_critic=self.target_critic,
                target_actor=self.target_actor,
                discount_factor=discount_factor,
                sample_num=self.sample_action_num,
                lr=3e-4
            )
            self.e_step = EStepContinuous(self.ds, self.da, self.target_actor, self.target_critic,
                                          self.sample_action_num)
            self.m_step = MStepContinuous(self.ds, self.da, self.actor, sample_action_num,
                                          alpha_mean_scale, alpha_var_scale,
                                          alpha_mean_max, alpha_var_max, kl_mean_constraint, kl_var_constraint,
                                          3e-4)
        else:
            self.critic_optimizer = CriticOptimizerDiscrete(
                critic=self.critic,
                target_critic=self.target_critic,
                target_actor=self.target_actor,
                discount_factor=discount_factor,
                lr=3e-4
            )
            self.e_step = EStepDiscrete(self.ds, self.da, self.target_actor, self.target_critic)
            self.m_step = MStepDiscrete(self.ds, self.da, self.actor, alpha_scale, alpha_max, kl_constraint, 3e-4)

        # Sampler
        self.sampler = SamplerSimple(env, sample_episode_maxstep)

        # Buffers
        self.experiences = ExperienceBuffer()

        self.iteration = 1

    def train(self, iteration_num=1000, log_dir='log'):
        """
        :param iteration_num:
        :param log_dir:
        """

        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.iteration, iteration_num + 1):

            print('iteration :', it)

            # Sample fresh episodes
            samples = self.sampler.sample(self.target_actor, self.sample_episode_num)
            replay = ReplayBuffer()
            replay.store_episodes(samples)
            self.experiences.store_episodes(samples)

            # Policy Evaluation
            # [2] 3 Policy Evaluation (Step 1)
            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(
                            SubsetRandomSampler(range(len(replay))), self.batch_size, drop_last=True),
                        desc='policy evaluation {}/{}'.format(r + 1, self.episode_rerun_num)):
                    # Load batch
                    state_batch, action_batch, next_state_batch, reward_batch = zip(*[replay[index] for index in indices])

                    # Critic optimization
                    loss_q, q = self.critic_optimizer.train(
                        batch_state=state_batch,
                        batch_action=action_batch,
                        batch_next_state=next_state_batch,
                        batch_reward=reward_batch,
                    )

            # Skip if there is not enough experiences
            if len(self.experiences) < self.batch_size * 8:
                continue

            # Policy Improvement (Step 2 + 3)
            for r in range(self.episode_rerun_num * 10):

                # Load experiences
                state_batch = self.experiences.sample(self.batch_size * 8).to(self.device)

                # E-Step of Policy Improvement
                qij, actions, probs = self.e_step.train(state_batch, self.ε_dual)

                # M-Step of Policy Improvement
                for _ in range(self.mstep_iteration_num):
                    self.m_step.train(state_batch, qij, actions, probs)

            #
            # Copy to target networks
            #

            self.__update_param()

            #
            # Logging
            #

            # mean_loss_q = np.mean(mean_loss_q)
            # mean_loss_p = np.mean(mean_loss_p)
            # mean_loss_l = np.mean(mean_loss_l)
            # mean_est_q = np.mean(mean_est_q)
            # # if self.continuous_action_space:
            # #     max_kl_μ = np.max(max_kl_μ)
            # #     max_kl_Σ = np.max(max_kl_Σ)
            # #     mean_Σ_det = np.mean(mean_Σ_det)
            # # else:  # discrete action space
            # #     max_kl = np.max(max_kl)
            print('iteration :', it)
            # print('  replay buffer :', buff_sz)
            print('  experience buffer :', len(self.experiences))
            # print('  mean return :', mean_return)
            # print('  mean reward :', mean_reward)
            # print('  mean loss_q :', mean_loss_q)
            # print('  mean loss_p :', mean_loss_p)
            # print('  mean loss_l :', mean_loss_l)
            # print('  mean est_q :', mean_est_q)
            # print('  η :', self.e_step.eta)
            # if self.continuous_action_space:
            #     # print('  max_kl_μ :', max_kl_μ)
            #     # print('  max_kl_Σ :', max_kl_Σ)
            #     # print('  mean_Σ_det :', mean_Σ_det)
            #     print('  α_μ :', self.α_μ)
            #     print('  α_Σ :', self.α_Σ)
            # else:  # discrete action space
            #     # print('  max_kl :', max_kl)
            #     print('  α :', self.α)
            #
            # writer.add_scalar('return', mean_return, it)
            # writer.add_scalar('reward', mean_reward, it)
            # writer.add_scalar('loss_q', mean_loss_q, it)
            # writer.add_scalar('loss_p', mean_loss_p, it)
            # writer.add_scalar('loss_l', mean_loss_l, it)
            # writer.add_scalar('mean_q', mean_est_q, it)
            # writer.add_scalar('η', self.e_step.eta, it)
            # if self.continuous_action_space:
            #     # writer.add_scalar('max_kl_μ', max_kl_μ, it)
            #     # writer.add_scalar('max_kl_Σ', max_kl_Σ, it)
            #     # writer.add_scalar('mean_Σ_det', mean_Σ_det, it)
            #     writer.add_scalar('α_μ', self.α_μ, it)
            #     writer.add_scalar('α_Σ', self.α_Σ, it)
            # else:
            #     # writer.add_scalar('η_kl', max_kl, it)
            #     writer.add_scalar('α', self.α, it)
            # writer.flush()

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
