import random
import torch
import torch.nn as nn


class DynamicsTrainer:
    """
    Training with Scheduled Sampling (decaying teacher forcing) to handle
    autoregressive error accumulation over K steps.
    """

    def __init__(self, model, lr=1e-3, epsilon_start=1.0, epsilon_end=0.1, decay_steps=10000):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.step_count = 0

    def epsilon(self):
        frac = min(self.step_count / self.decay_steps, 1.0)
        return self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)

    def train_step(self, batch_seq):
        self.model.train()
        self.step_count += 1
        eps = self.epsilon()

        if random.random() < eps:
            loss = self.model.dynamics_loss(batch_seq)
        else:
            seq = batch_seq[:, :-1, :]
            target = batch_seq[:, 1:, :]
            pred = self.model.forward(seq)
            sampled = pred.detach()
            mixed = torch.where(torch.rand_like(seq) < 0.5, seq, sampled)
            pred2 = self.model.forward(mixed)
            loss = nn.functional.mse_loss(pred2[:, -1:, :], target[:, -1:, :]) + nn.functional.mse_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, dataloader):
        total = 0
        for batch in dataloader:
            total += self.train_step(batch)
        return total / len(dataloader)