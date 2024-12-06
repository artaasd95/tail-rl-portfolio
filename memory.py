import torch
import numpy as np

class PrioritizedMemory:
    def __init__(self, capacity, prob_alpha=0.6):
        self.capacity = capacity
        self.prob_alpha = prob_alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, value, done, error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, value, done))
        else:
            self.buffer[self.position] = (state, action, reward, value, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.prob_alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        weights = torch.tensor(weights, dtype=torch.float32)
        batch = [(x[0], x[1], x[2], x[3], x[4]) for x in samples]

        states, actions, rewards, values, dones = zip(*batch)
        batch = (states, actions, rewards, values, dones)

        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority