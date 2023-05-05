import numpy as np
from collections import deque

import torch


class ExperienceBuffer:
    def __init__(self, buffer_size=1000000):
        self.buffer = deque(maxlen=buffer_size)

    def store_episodes(self, episodes):
        for ep in episodes:
            for exp in ep:
                self.buffer.append(exp[0])  # First one is state

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return torch.from_numpy(np.stack(batch)).type(torch.float32)

    def __len__(self):
        return len(self.buffer)
