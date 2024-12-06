import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np



class TransformerCore(nn.Module):
    def __init__(self, state_size):
        super(TransformerCore, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_size, nhead=4),
            num_layers=2
        )

    def forward(self, x):
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        return x
    


class Actor(nn.Module):
    def __init__(self, state_size, n_assets):
        super(Actor, self).__init__()
        self.core = TransformerCore(state_size)
        self.actor_fc = nn.Linear(state_size, n_assets)

    def forward(self, x):
        x = self.core(x)
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        return action_probs



class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.core = TransformerCore(state_size)
        self.critic_fc = nn.Linear(state_size, 1)

    def forward(self, x):
        x = self.core(x)
        state_value = self.critic_fc(x)
        return state_value



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns



def train_agent(env, actor, critic, actor_optimizer, critic_optimizer, memory,
                num_episodes=100, batch_size=32, gamma=0.99, tau=0.95):
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = state.unsqueeze(0)

            # Actor selects an action
            action_probs = actor(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done = env.step(action.detach().numpy())

            # Critic evaluates the state
            state_value = critic(state_tensor)
            next_state_value = critic(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))

            # Store in prioritized memory
            error = abs(reward + gamma * next_state_value.item() - state_value.item())
            memory.add(state, action, reward, state_value.item(), done, error)

            state = next_state

        # Sample from prioritized memory
        if len(memory.buffer) >= batch_size:
            batch, indices, weights = memory.sample(batch_size)
            states, actions, rewards, values, dones = batch

            # Compute returns and advantages
            returns = compute_gae(next_state_value.item(), rewards, [1 - d for d in dones], values, gamma, tau)
            advantages = torch.tensor(returns) - torch.tensor(values).squeeze().detach()

            # Update actor-critic
            for idx, (state, action, value, ret, advantage) in enumerate(zip(states, actions, values, returns, advantages)):
                state_tensor = state.unsqueeze(0)

                # Actor update
                action_probs = actor(state_tensor)
                action_dist = Categorical(action_probs)
                log_prob = action_dist.log_prob(action)
                actor_loss = -(log_prob * advantage.detach() * weights[idx])

                # Critic update
                state_value = critic(state_tensor)
                critic_loss = F.mse_loss(state_value.squeeze(), ret * weights[idx])

                # Total loss
                loss = actor_loss + critic_loss

                # Update priorities
                memory.update_priorities(indices, (advantage ** 2 + 1e-5).numpy())

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

        print(f"Episode {episode + 1} finished.")
