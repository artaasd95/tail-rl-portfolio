import torch
import torch.nn.functional as F


def calculate_expected_shortfall(allocations, historical_returns, alpha=0.95):
    """
    Calculate the Expected Shortfall (ES) using historical returns data.

    :param allocations: torch.Tensor, portfolio allocations for each asset
    :param historical_returns: torch.Tensor, historical returns data (num_periods x num_assets)
    :param alpha: float, confidence level for VaR/ES calculation
    :return: torch.Tensor, calculated Expected Shortfall
    """
    # Calculate portfolio returns from historical data
    portfolio_returns = torch.matmul(historical_returns, allocations)

    # Determine VaR at (1-alpha) quantile
    var_threshold = torch.quantile(portfolio_returns, 1 - alpha)

    # Calculate excess returns beyond VaR
    excess_returns = portfolio_returns[portfolio_returns > var_threshold] - var_threshold

    # Estimate ES as the mean of excess returns
    if len(excess_returns) > 0:
        expected_shortfall = excess_returns.mean()
    else:
        expected_shortfall = torch.tensor(0.0)

    return expected_shortfall



def reward_function(allocation, price_relatives, max_return, min_return, 
                    max_dev, min_dev, gamma=0.1):
    log_returns = torch.log(torch.dot(allocation, price_relatives))
    avg_return = (max_return + min_return) / 2

    # Normalizing portfolio returns
    norm_return = 0.5 * (log_returns - avg_return) / (max_return - min_return)
    
    # Calculate power-law deviation
    power_law_deviation = torch.norm(allocation - allocation.pow(2).mean())  # Simplified deviation
    norm_deviation = 0.5 * (max_dev - power_law_deviation) / (max_dev - min_dev)
    
    # Calculate expected shortfall or tail risk part
    es = calculate_expected_shortfall(allocation, price_relatives)
    norm_tail_risk = gamma * es / 0.5  # Assume max ES around 0.5
    
    # Final reward, should scale between -1 and 1
    reward = norm_return + norm_deviation - norm_tail_risk
    return torch.clamp(reward, -1.0, 1.0)