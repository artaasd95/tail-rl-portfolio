import yfinance as yf
import torch
from reward import reward_function

class FinanceEnv:
    def __init__(self, data):
        self.n_assets = len(data.columns)
        self.current_step = 0
        self.max_steps = len(self.data)

    def reset(self):
        self.current_step = 0
        return torch.tensor(self.data[self.current_step], dtype=torch.float32)

    def step(self, allocations):
        if self.current_step >= self.max_steps - 1:
            done = True
            reward = 0
        else:
            self.current_step += 1
            done = False
            prev_prices = self.data[self.current_step - 1]
            current_prices = self.data[self.current_step]
            price_relatives = current_prices / prev_prices
            reward = self._calculate_reward(allocations, price_relatives)
        
        next_state = torch.tensor(self.data[self.current_step], dtype=torch.float32)
        return next_state, reward, done

    def _calculate_reward(self, allocations, price_relatives):
        reward = reward_function(allocations, price_relatives)
        return reward.item()