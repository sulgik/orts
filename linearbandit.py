from typing import Dict, List, Optional, Union
import numpy as np


class LinearBandit(object):
    """
    Linear (Gaussian) Bandit implementation with Thompson Sampling.

    This class implements Thompson Sampling for multi-armed bandits with
    continuous (Gaussian) rewards. Each action's mean reward is modeled
    as a Gaussian distribution, and the posterior is updated using
    Bayesian inference with conjugate Gaussian priors.

    Parameters
    ----------
    mu : dict, optional
        Dictionary mapping action names to their prior mean rewards.
        Default is None (will be initialized from first observations).
    sigma : dict, optional
        Dictionary mapping action names to their prior standard deviations.
        Default is None (will use default_sigma for all actions).
    default_sigma : float, optional
        Default prior standard deviation for new actions.
        Default is 1.0.
    obs_noise : float, optional
        Assumed observation noise (standard deviation of rewards).
        Default is 1.0.

    Attributes
    ----------
    mu : dict
        Current posterior mean for each action
    sigma : dict
        Current posterior standard deviation for each action
    counts : dict
        Number of observations for each action
    sum_rewards : dict
        Sum of observed rewards for each action
    default_sigma : float
        Default prior sigma for new actions
    obs_noise : float
        Observation noise standard deviation

    Examples
    --------
    >>> bandit = LinearBandit()
    >>> obs = {"action_1": [1.2, 1.5, 1.3], "action_2": [0.8, 0.9]}
    >>> bandit.update(obs)
    >>> bandit.win_prop()
    {'action_1': 0.892, 'action_2': 0.108}
    """

    def __init__(self,
                 mu: Optional[Dict[str, float]] = None,
                 sigma: Optional[Dict[str, float]] = None,
                 default_sigma: float = 1.0,
                 obs_noise: float = 1.0) -> None:
        """Initialize LinearBandit with optional prior parameters."""
        if default_sigma <= 0:
            raise ValueError("default_sigma must be positive, got {}".format(default_sigma))
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive, got {}".format(obs_noise))

        self.mu: Dict[str, float] = mu.copy() if mu is not None else {}
        self.sigma: Dict[str, float] = sigma.copy() if sigma is not None else {}
        self.counts: Dict[str, int] = {action: 0 for action in self.mu.keys()}
        self.sum_rewards: Dict[str, float] = {action: 0.0 for action in self.mu.keys()}
        self.default_sigma = default_sigma
        self.obs_noise = obs_noise

    def get_models(self) -> List[str]:
        """
        Get the list of currently tracked actions.

        Returns
        -------
        list
            List of action names currently being tracked.
        """
        return list(self.mu.keys())

    def update(self,
               obs: Dict[str, Union[List[float], float]],
               decay: float = 0.0) -> None:
        """
        Update the bandit model with new observations.

        For each action, observations are continuous reward values.
        The posterior is updated using Bayesian inference with Gaussian
        conjugate priors.

        Parameters
        ----------
        obs : dict
            Dictionary mapping action names to rewards.
            Values can be:
            - A list of rewards: [1.2, 1.5, 1.3]
            - A single reward value: 1.4
        decay : float, optional
            Decay parameter for discounting prior information (0.0 to 1.0).
            Higher values give more weight to new observations.
            Default is 0.0 (no decay).

        Returns
        -------
        None
            Updates self.mu, self.sigma, self.counts, self.sum_rewards in place.

        Raises
        ------
        ValueError
            If obs is empty or decay is out of range.

        Notes
        -----
        The Bayesian update for Gaussian conjugate prior:

        Prior: N(mu_0, sigma_0^2)
        Likelihood: rewards ~ N(mu, obs_noise^2)
        Posterior: N(mu_n, sigma_n^2)

        Where:
        - precision_0 = 1 / sigma_0^2
        - precision_obs = n / obs_noise^2
        - precision_n = precision_0 + precision_obs
        - sigma_n = 1 / sqrt(precision_n)
        - mu_n = (precision_0 * mu_0 + precision_obs * mean_reward) / precision_n
        """
        # Input validation
        if not obs:
            raise ValueError("obs dictionary cannot be empty")

        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0.0 and 1.0, got {}".format(decay))

        for action, rewards in obs.items():
            # Convert single reward to list
            if isinstance(rewards, (int, float)):
                reward_list = [float(rewards)]
            elif isinstance(rewards, (list, tuple, np.ndarray)):
                reward_list = [float(r) for r in rewards]
            else:
                raise ValueError(
                    "Rewards must be a number, list, tuple, or array, got {} for action '{}'".format(
                        type(rewards), action
                    )
                )

            if len(reward_list) == 0:
                continue

            # Initialize action if new
            if action not in self.mu:
                self.mu[action] = 0.0
                self.sigma[action] = self.default_sigma
                self.counts[action] = 0
                self.sum_rewards[action] = 0.0

            # Apply decay to prior if specified
            if decay > 0:
                # Decay increases uncertainty and pulls mean toward 0
                self.sigma[action] = np.sqrt(
                    (1 - decay) * self.sigma[action]**2 + decay * self.default_sigma**2
                )
                self.mu[action] = (1 - decay) * self.mu[action]

            # Get current prior parameters
            mu_0 = self.mu[action]
            sigma_0 = self.sigma[action]

            # Compute statistics from new observations
            n = len(reward_list)
            mean_reward = np.mean(reward_list)

            # Bayesian update: Gaussian conjugate prior
            # Prior precision (inverse variance)
            precision_0 = 1.0 / (sigma_0 ** 2)

            # Likelihood precision (from n observations)
            precision_obs = n / (self.obs_noise ** 2)

            # Posterior precision
            precision_n = precision_0 + precision_obs

            # Posterior mean (precision-weighted average)
            mu_n = (precision_0 * mu_0 + precision_obs * mean_reward) / precision_n

            # Posterior standard deviation
            sigma_n = 1.0 / np.sqrt(precision_n)

            # Update parameters
            self.mu[action] = mu_n
            self.sigma[action] = sigma_n
            self.counts[action] += n
            self.sum_rewards[action] += sum(reward_list)

    def win_prop(self,
                 action_list: Optional[List[str]] = None,
                 draw: int = 100000,
                 aggressive: float = 1.0) -> Dict[str, float]:
        """
        Calculate the winning probability for each action using Thompson Sampling.

        Uses Monte Carlo simulation to estimate the probability that each action
        has the highest mean reward.

        Parameters
        ----------
        action_list : list, optional
            List of actions to consider. If None, uses all tracked actions.
            Default is None.
        draw : int, optional
            Number of Monte Carlo samples to draw. More samples give more
            accurate estimates but take longer to compute.
            Default is 100000.
        aggressive : float, optional
            Aggressiveness parameter for probability adjustment (gamma exponent).
            Values > 1 increase exploitation, values < 1 increase exploration.
            Default is 1.0 (no adjustment).

        Returns
        -------
        dict
            Dictionary mapping action names to their winning probabilities.
            Probabilities sum to 1.0.

        Raises
        ------
        ValueError
            If draw is not positive or aggressive is not positive.

        Examples
        --------
        >>> bandit.win_prop()
        {'action_1': 0.67, 'action_2': 0.33}
        >>> bandit.win_prop(aggressive=2.0)  # More exploitation
        {'action_1': 0.82, 'action_2': 0.18}
        """
        # Input validation
        if draw <= 0:
            raise ValueError("draw must be positive, got {}".format(draw))

        if aggressive <= 0:
            raise ValueError("aggressive must be positive, got {}".format(aggressive))

        if action_list is None:
            action_list = list(self.mu.keys())

        if len(action_list) == 0:
            return {}
        elif len(action_list) == 1:
            return {action_list[0]: 1.0}

        # Split into observed and unobserved actions
        action_list_observed = [a for a in action_list if a in self.mu]
        action_list_unobserved = [a for a in action_list if a not in self.mu]

        n_observed = len(action_list_observed)
        n_unobserved = len(action_list_unobserved)

        out = {}

        if n_observed == 0:
            # All actions are unobserved, uniform distribution
            for action in action_list_unobserved:
                out[action] = 1.0 / n_unobserved
        elif n_observed == 1:
            # Only one observed action
            out[action_list_observed[0]] = 1.0 / (1 + n_unobserved)
        else:
            # Multiple observed actions: run Monte Carlo simulation
            # Sample from each action's posterior distribution
            samples = np.zeros((draw, n_observed))
            for i, action in enumerate(action_list_observed):
                samples[:, i] = np.random.normal(
                    self.mu[action],
                    self.sigma[action],
                    draw
                )

            # Count how many times each action wins
            counts = np.zeros(n_observed)
            winner_idxs = np.argmax(samples, axis=1)
            for idx in winner_idxs:
                counts[idx] += 1

            # Apply aggressive parameter
            count_gamma = counts ** aggressive
            p_winner = count_gamma / np.sum(count_gamma)

            # Assign probabilities accounting for unobserved actions
            for i, action in enumerate(action_list_observed):
                out[action] = p_winner[i] * n_observed / (n_observed + n_unobserved)

        # Unobserved actions get uniform probability
        for action in action_list_unobserved:
            out[action] = 1.0 / (n_observed + n_unobserved)

        return out

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get current statistics for all actions.

        Returns
        -------
        dict
            Dictionary mapping action names to their statistics:
            - 'mu': posterior mean
            - 'sigma': posterior standard deviation
            - 'count': number of observations
            - 'mean_reward': empirical mean of observed rewards
        """
        stats = {}
        for action in self.mu.keys():
            stats[action] = {
                'mu': self.mu[action],
                'sigma': self.sigma[action],
                'count': self.counts[action],
                'mean_reward': (
                    self.sum_rewards[action] / self.counts[action]
                    if self.counts[action] > 0 else 0.0
                )
            }
        return stats
