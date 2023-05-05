import torch
from mpo import MPO
import gymnasium as gym


def main():
    # Loading the environment
    device = ("cuda" if torch.cuda.is_available() else "cpu")  # MPS is not supported
    print(f"Using {device} device")

    # Create environment
    torch.autograd.set_detect_anomaly(True)
    env = gym.make("LunarLander-v2", continuous=True)
    model = MPO(device, env, sample_episode_maxstep=200)

    # Train
    model.train(iteration_num=200)

    # Test
    env = gym.make("LunarLander-v2", continuous=True, render_mode="human")
    while True:
        observation, info = env.reset()
        while True:

            # Execute action
            action = model.actor.action(observation)

            # Render
            observation, reward, terminated, truncated, info = env.step(action)

            # Break
            if terminated or truncated:
                break


if __name__ == '__main__':
    main()
