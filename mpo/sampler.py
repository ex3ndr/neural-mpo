class SamplerSimple:
    def __init__(self, env, max_step):
        self.env = env
        self.max_step = max_step

    def sample(self, actor, episodes):
        result = []
        for i in range(episodes):
            result.append(self.__sample_episode(actor))
        return result

    def __sample_episode(self, actor):
        buff = []
        state, _ = self.env.reset()
        for steps in range(self.max_step):
            action = actor.action_sample(state)
            next_state, reward, termination, _, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))
            if termination:
                break
            else:
                state = next_state
        return buff
