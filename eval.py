import torch
from mpo import MPO
import gymnasium as gym
import time

def main():
    # Loading the environment
    device = ("cuda" if torch.cuda.is_available() else "cpu")  # MPS is not supported
    print(f"Using {device} device")

    # Create environment
    env = gym.make("HalfCheetah-v4", render_mode="human")
    model = MPO(device, env)
    while True:
        observation, info = env.reset()
        model.load_model("latest.ph")
        total_reward = 0
        max_t = 200
        t = 0
        while t < max_t:
            t = t + 1

            # Execute action
            action = model.act_sample(observation)

            # print(action)

            # Render
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Break
            if terminated or truncated:
                break

        print(total_reward)


if __name__ == '__main__':
    main()
