import torch
from mpo import MPO
import gymnasium as gym


def main():
    # Loading the environment
    device = ("cuda" if torch.cuda.is_available() else "cpu")  # MPS is not supported
    print(f"Using {device} device")

    # Create environment
    env = gym.make("HalfCheetah-v4")
    model = MPO(device, env, sample_episode_max_step=1024, sample_episode_num=16, sync_to="latest.ph")

    # Train
    model.train(iteration_num=200, log_dir="runs/cheetah_1")


if __name__ == '__main__':
    main()
