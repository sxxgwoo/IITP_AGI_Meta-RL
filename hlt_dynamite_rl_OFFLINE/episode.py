class Episode:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dts = []

    def add(self, obs, action, reward, dt):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dts.append(dt)