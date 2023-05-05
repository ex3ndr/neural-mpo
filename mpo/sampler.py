from tqdm import tqdm


class SamplerSimple:
    def __init__(self, env, max_step):
        self.env = env
        self.max_step = max_step

    def sample(self, actor, episodes):
        result = []
        total_steps = 0
        for _ in tqdm(range(episodes), desc='Sampling'):
            sampled = self.__sample_episode(actor)
            result.append(sampled)
            total_steps += len(sampled)
        return result, total_steps

    def __sample_episode(self, actor):
        buff = []
        state, _ = self.env.reset()
        for steps in range(self.max_step):
            action = actor.action_sample(state)
            next_state, reward, termination, _, _ = self.env.step(action)
            # reward = reward * 0.001  # Rescale reward
            buff.append((state, action, next_state, reward))
            if termination:
                break
            else:
                state = next_state
        return buff


class SamplerBalanced:
    def __init__(self, env, max_step):
        self.env = env
        self.max_step = max_step

    def sample(self, actor, episodes):
        result = []
        buff = []
        state, _ = self.env.reset()
        with tqdm(total=episodes * self.max_step, desc='Sampling') as pbar:
            for steps in range(self.max_step * episodes):
                action = actor.action_sample(state)
                next_state, reward, termination, truncated, _ = self.env.step(action)
                buff.append((state, action, next_state, reward))
                if termination or truncated:
                    state, _ = self.env.reset()
                    result.append(buff)
                    buff = []
                else:
                    state = next_state
                pbar.update(1)
        if len(buff) > 0:
            result.append(buff)
        return result, episodes * self.max_step
