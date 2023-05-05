import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from mpo.actor import ActorContinuous, ActorDiscrete
from mpo.critic import Critic, CriticOptimizerContinuous, CriticOptimizerDiscrete
from mpo.e_step import EStepContinuous, EStepDiscrete
from mpo.experience import ExperienceBuffer
from mpo.m_step import MStepDiscrete, MStepContinuous
from mpo.replaybuffer import ReplayBuffer
from mpo.sampler import SamplerSimple, SamplerBalanced


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
                 kl_mean_constraint=0.001,
                 kl_var_constraint=0.00001,
                 kl_constraint=0.01,
                 discount_factor=0.99,
                 alpha_mean_scale=1.0,
                 alpha_var_scale=1.0,
                 alpha_scale=1.0,
                 alpha_mean_max=1.0,
                 alpha_var_max=1.0,
                 alpha_max=1.0,

                 # Sampling
                 sample_episode_num=200,
                 sample_episode_max_step=200,
                 sample_action_num=32,

                 # Batching
                 eval_batch_size=64,
                 eval_batch=4,
                 improve_batch=8,
                 improve_batch_size=256,
                 improve_m_step_count=64,

                 # Auto save
                 sync_to=None
                 ):
        self.device = device
        self.env = env
        self.sync_to = sync_to

        # Load the environment parameter space
        self.observations = env.observation_space.shape[0]
        if self.env.action_space.dtype == np.float32:
            self.continuous_action_space = True
            self.actions = env.action_space.shape[0]
        else:
            self.continuous_action_space = False
            self.actions = env.action_space.n

        # Sampling
        self.sample_episode_num = sample_episode_num
        self.sample_episode_max_step = sample_episode_max_step
        self.sample_action_num = sample_action_num

        # Batching
        self.eval_batch_size = eval_batch_size
        self.eval_batch = eval_batch
        self.improve_batch = improve_batch
        self.improve_batch_size = improve_batch_size
        self.improve_m_step_count = improve_m_step_count

        # Networks
        self.critic = Critic(self.device, self.observations, self.actions).to(self.device)
        self.target_critic = Critic(self.device, self.observations, self.actions).to(self.device)
        if self.continuous_action_space:
            self.actor = ActorContinuous(self.device, self.observations, self.actions, env.action_space).to(self.device)
            self.target_actor = ActorContinuous(self.device, self.observations, self.actions, env.action_space).to(
                self.device)
        else:
            self.actor = ActorDiscrete(self.device, self.observations, self.actions).to(self.device)
            self.target_actor = ActorDiscrete(self.device, self.observations, self.actions).to(self.device)
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
            self.e_step = EStepContinuous(self.observations, self.actions, self.target_actor, self.target_critic,
                                          dual_constraint, self.sample_action_num)
            self.m_step = MStepContinuous(self.observations, self.actions, self.actor, sample_action_num,
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
            self.e_step = EStepDiscrete(self.observations, self.actions, self.target_actor, self.target_critic,
                                        dual_constraint)
            self.m_step = MStepDiscrete(self.observations, self.actions, self.actor, alpha_scale, alpha_max,
                                        kl_constraint, 3e-4)

        # Sampler
        # self.sampler = SamplerSimple(env, sample_episode_max_step)
        self.sampler = SamplerBalanced(env, sample_episode_max_step)

        # Buffers
        self.experiences = ExperienceBuffer()

        # Current iteration
        self.iteration = 0
        self.total_steps = 0
        self.total_episodes = 0

        # Load model
        if self.sync_to is not None:
            if os.path.exists(sync_to):
                self.load_model(self.sync_to)

    def act(self, state):
        return self.target_actor.action(state)

    def act_sample(self, state):
        return self.target_actor.action_sample(state)

    def train(self, iteration_num=1000, log_dir=None):
        """
        :param iteration_num:
        :param log_dir:
        """

        writer = SummaryWriter(log_dir)
        for _ in range(iteration_num):
            self.iteration += 1
            it = self.iteration

            # Sample fresh episodes
            samples, steps = self.sampler.sample(self.target_actor, self.sample_episode_num)
            replay = ReplayBuffer()
            replay.store_episodes(samples)
            self.experiences.store_episodes(samples)
            self.total_steps += steps
            self.total_episodes += len(samples)

            # Policy Evaluation
            for _ in tqdm(range(self.eval_batch), desc='Policy Evaluation'):
                replay_sampler = BatchSampler(SubsetRandomSampler(range(len(replay))), self.eval_batch_size,
                                              drop_last=True)
                for indices in replay_sampler:
                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[replay[index] for index in indices])

                    # Critic optimization
                    self.critic_optimizer.train(
                        batch_state=state_batch,
                        batch_action=action_batch,
                        batch_next_state=next_state_batch,
                        batch_reward=reward_batch,
                    )

            # Copy critic to target critic
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)

            # Skip if there is not enough experiences
            if len(self.experiences) < self.improve_batch_size:
                continue

            # Policy Improvement
            for _ in tqdm(range(self.improve_batch), desc='Policy Improvement'):

                # Load experiences
                state_batch = self.experiences.sample(self.improve_batch_size).to(self.device)

                # E-Step of Policy Improvement
                qij, actions, probs = self.e_step.train(state_batch)

                # M-Step of Policy Improvement
                for _ in range(self.improve_m_step_count):
                    self.m_step.train(state_batch, qij, actions, probs)

            #
            # Copy actor to target actor
            #

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

            #
            # Calculate statistics
            #

            stat_mean_return = replay.mean_return()
            stat_mean_reward = replay.mean_reward()

            #
            # Logging
            #

            print('iteration:', it)
            print('  episodes    :', self.total_episodes)
            print('  steps       :', self.total_steps)
            print('  mean return :', stat_mean_return)
            print('  mean reward :', stat_mean_reward)
            print('  eta         :', self.e_step.eta)
            writer.add_scalar('train/mean_return', stat_mean_return, it)
            writer.add_scalar('train/mean_reward', stat_mean_reward, it)
            writer.add_scalar('train/eta', self.e_step.eta, it)

            #
            # Save model
            #
            if self.sync_to is not None:
                self.save_model(self.sync_to)

        # end training
        if writer is not None:
            writer.close()

    def save_model(self, path):
        data = {
            # Parameters
            'iteration': self.iteration,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,

            # Models
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),

            # Optimizers
            'actor_optim_state_dict': self.m_step.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.critic_optimizer.state_dict()
        }
        torch.save(data, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.iteration = checkpoint['iteration']
        self.total_episodes = checkpoint['total_episodes']
        self.total_steps = checkpoint['total_steps']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.m_step.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()
