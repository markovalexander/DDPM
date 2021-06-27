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

    def update_with_all_losses(self, ts, losses):
        pass


class UniformSampler(AbstractSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossSecondMomentResampler(AbstractSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float32)

        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


def get_time_sampler(sampler_type):
    if sampler_type == "LOSS-SECOND-MOMENT":
        return LossSecondMomentResampler
    else:
        return UniformSampler
