from abc import ABC, abstractmethod

import torch
import numpy as np


class AbstractSampler(ABC):
    @abstractmethod
    def weights(self):
        ...

    def sample(self, batch_size, device):
        weights = self.weights()
        probs = weights / np.sum(weights)
        indxes = np.random.choice(len(probs), size=(batch_size,), p=probs)
        time = torch.from_numpy(indxes).to(device=device, dtype=torch.int64)
        weights = 1 / (len(probs) * probs[indxes])
        weights = torch.from_numpy(weights).to(device=device, dtype=torch.float32)
        return time, weights


class UniformSampler(AbstractSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights
