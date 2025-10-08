class RewardNormalizer:
    def __init__(self, eps=1e-8):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0
        self.eps = eps

    def update(self, reward: float):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2

    def normalize(self, reward: float):
        std = (self.var / max(self.count, 1) + self.eps) ** 0.5
        return (reward - self.mean) / std
