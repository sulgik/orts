from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy import fill_diagonal
from numpy.linalg import pinv, inv

from utils import estimate, is_pos_semidef


class LogisticBandit(object):
    """
    Logistic Bandit implementation with Thompson Sampling.

    Supports both Full-TS (Full Rank Thompson Sampling) and ORTS (Odds Ratio
    Thompson Sampling) for multi-armed bandit problems with logistic model.
    ORTS is more robust to time-varying effects compared to Full-TS.

    Parameters
    ----------
    mu : numpy.ndarray, optional
        Mean vector of the posterior distribution. Default is None.
    sigma_inv : numpy.ndarray, optional
        Inverse covariance matrix of the posterior distribution. Default is None.
    action_list : list, optional
        List of action names. Default is None.

    Attributes
    ----------
    mu : numpy.ndarray
        Current mean vector of posterior distribution
    sigma_inv : numpy.ndarray
        Current inverse covariance matrix of posterior distribution
    action_list : list
        Current list of action names

    Examples
    --------
    >>> orpar = LogisticBandit()
    >>> obs = {"arm_1": [30000, 300], "arm_2": [30000, 290]}
    >>> orpar.update(obs)
    >>> orpar.win_prop()
    """
    def __init__(self, mu: Optional[np.ndarray] = None,
                 sigma_inv: Optional[np.ndarray] = None,
                 action_list: Optional[List[str]] = None) -> None:
        self.mu, self.sigma_inv, self.action_list = None, None, None
        self._initialize(mu, sigma_inv, action_list)

    def _initialize(self, mu: Optional[np.ndarray],
                    sigma_inv: Optional[np.ndarray],
                    action_list: Optional[List[str]]) -> None:
        self.mu = np.copy(mu) if mu is not None else np.array([])
        self.sigma_inv = np.copy(sigma_inv) if sigma_inv is not None else np.empty((0, 0))
        self.action_list = action_list.copy() if action_list is not None else []

    def get_models(self) -> List[str]:
        """
        Get the list of currently tracked actions.

        Returns
        -------
        list
            List of action names currently being tracked.
        """
        return self.action_list

    def get_par(self, action_list: Optional[List[str]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get parameters (mu, sigma_inv) transformed for a specific action list.

        This method transforms the internal representation to use a different
        reference action or subset of actions.

        Parameters
        ----------
        action_list : list, optional
            List of actions to get parameters for. The last action in the list
            is used as the reference. Default is None.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray) or (None, None)
            Transformed (mu, sigma_inv) for the given action_list.
            Returns (None, None) if action_list is empty or None.
        """
        if not action_list:
            return None, None

        action_nonref = action_list[:-1]
        action_ref = action_list[-1]

        # transform_mat1: to new odds ratio    
        transform_mat1 = np.zeros((len(action_list), len(self.action_list)))
        transform_mat1[:, self.action_list.index(action_ref)]=-1
        for i, new_nonref in enumerate(action_nonref):
            for j, old_nonref in enumerate(self.action_list):
                if new_nonref == old_nonref:
                    transform_mat1[i,j]=1

        transform_mat1[-1, self.action_list.index(action_ref)]=1
        
        # transform_mat2: inverse from old odds ratios
        transform_mat2 = np.zeros((len(self.action_list), len(self.action_list)))
        transform_mat2[:,-1] = 1
        fill_diagonal(transform_mat2, 1)

        transform_mat = transform_mat1.dot(transform_mat2)

        mu = np.matmul(transform_mat, self.mu)

        tr_inv = np.rint(pinv(transform_mat))
        sigma_inv = tr_inv.T.dot(self.sigma_inv).dot(tr_inv)

        return mu, sigma_inv

    def transform(self, action_list: List[str]) -> None:
        mu, sigma_inv = self.get_par(action_list)
        self.__init__(
            mu = mu, sigma_inv = sigma_inv, action_list = action_list)

    def update(
        self, obs: Dict[str, List[float]], odds_ratios_only: bool = True,
        remove_not_observed: bool = False, decay: float = 0.0) -> None:
        """
        Update the bandit model with new observations.

        Parameters
        ----------
        obs : dict
            Dictionary mapping action names to [total_count, success_count].
            Example: {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        odds_ratios_only : bool, optional
            If True, use ORTS (Odds Ratio Thompson Sampling).
            If False, use Full-TS (Full Rank Thompson Sampling).
            Default is True.
        remove_not_observed : bool, optional
            If True, remove actions that were not observed.
            Default is False.
        decay : float, optional
            Decay parameter for discounting prior information (0.0 to 1.0).
            Higher values give more weight to new observations.
            Default is 0.0 (no decay).

        Returns
        -------
        None
            Updates self.mu, self.sigma_inv, and self.action_list in place.

        Raises
        ------
        ValueError
            If obs is empty, decay is out of range, or observation format is invalid.
        """
        # Input validation
        if not obs:
            raise ValueError("obs dictionary cannot be empty")

        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0.0 and 1.0, got {}".format(decay))

        # Validate observation format
        for action, values in obs.items():
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(
                    "Each observation must be [total_count, success_count], "
                    "got {} for action '{}'".format(values, action)
                )
            total, success = values
            if total < 0:
                raise ValueError(
                    "Total count must be non-negative, got {} for action '{}'".format(total, action)
                )
            if success < 0:
                raise ValueError(
                    "Success count must be non-negative, got {} for action '{}'".format(success, action)
                )
            if success > total:
                raise ValueError(
                    "Success count ({}) cannot exceed total count ({}) for action '{}'".format(
                        success, total, action
                    )
                )

        obs_valid = {i:obs[i] for i in obs.keys() if obs[i][0] > 0}

        if len(obs_valid) > 0:

            total_number=0.
            for value in obs_valid.values():
                total_number += value[0]

            # find arrangement
            action_on=[]
            action_newcome=[]

            for action_obs in obs_valid.keys():
                if action_obs in self.get_models():                
                    action_on.append(action_obs)
                else: 
                    action_newcome.append(action_obs)

            action_nonobserved = [x for x in self.get_models() if x not in obs_valid.keys()]
            action_list = action_newcome + action_nonobserved + action_on

            prior = self.get_par(action_nonobserved + action_on)
            if (prior[0] is not None) and odds_ratios_only:
                if len(prior[0]) > 1:
                    sigma_inv_new = np.zeros((len(prior[0]), len(prior[0])))
                    sigma_inv_new[:-1,:-1] = inv(inv(prior[1])[:-1,:-1])
                    prior = (prior[0], sigma_inv_new)

            obs_list = [obs_valid[action] for action in action_newcome + action_on]
            index = [len(action_newcome), len(action_newcome + action_nonobserved), len(action_list)]

            # estimate part    
            parameters = estimate(prior, obs_list, index, discount = decay)
                            
            self.mu = parameters[0]
            self.sigma_inv = parameters[1]
            self.action_list = action_list

    def win_prop(self, action_list: Optional[List[str]] = None,
                 draw: int = 100000, aggressive: float = 1.0) -> Dict[str, float]:
        """
        Calculate the winning probability for each action using Thompson Sampling.

        Uses Monte Carlo simulation to estimate the probability that each action
        has the highest reward.

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
        >>> orpar.win_prop()
        {'arm_1': 0.543, 'arm_2': 0.457}
        >>> orpar.win_prop(aggressive=2.0)  # More aggressive exploitation
        {'arm_1': 0.687, 'arm_2': 0.313}
        """
        # Input validation
        if draw <= 0:
            raise ValueError("draw must be positive, got {}".format(draw))

        if aggressive <= 0:
            raise ValueError("aggressive must be positive, got {}".format(aggressive))

        if action_list is None:
            action_list = self.action_list

        if len(action_list) == 0:
            return {}
        elif len(action_list) == 1:
            return {action_list[0]: 1.}

        # split query
        action_list_observed   = list()
        action_list_unobserved = list()

        for x in action_list:
            if x in self.action_list:
                action_list_observed.append(x)
            else:
                action_list_unobserved.append(x)

        n_observed   = len(action_list_observed  )
        n_unobserved = len(action_list_unobserved)

        out = dict()
        if n_observed == 1:
            out[action_list_observed[0]] = 1. /(1 + n_unobserved)
        elif n_observed > 1:
            mu, sigma_inv = self.get_par(action_list_observed)
            
            sigma = pinv(sigma_inv)
            
            if len(sigma) == 2:
                mc = np.random.normal(mu[0], np.sqrt(sigma[0,0]), draw)
                mc = mc.reshape(draw,1)
            else:
                # Ensure positive semidefinite covariance matrix
                max_psd_iterations = 100
                psd_increment = 0.001
                iteration = 0
                while not is_pos_semidef(sigma[:-1,:-1]) and iteration < max_psd_iterations:
                    print("Warning: not positive semidefinite, adjusting diagonal")
                    np.fill_diagonal(sigma, sigma.diagonal() + psd_increment)
                    iteration += 1

                if not is_pos_semidef(sigma[:-1,:-1]):
                    raise ValueError(
                        "Failed to make covariance matrix positive semidefinite after {} iterations".format(
                            max_psd_iterations
                        )
                    )

                mc = np.random.multivariate_normal(mu[:-1], sigma[:-1,:-1], draw)

            # concatenate
            mc = np.concatenate((mc, np.zeros([draw, 1])), axis = 1)

            # count frequency of each arm being winner 
            counts = [0 for _ in range(len(mu))]
            winner_idxs = np.asarray(mc.argmax(axis = 1)).reshape(draw, )
            for idx in winner_idxs:
                counts[idx] += 1
            
            # divide by draw to approximate probability distribution
            count_gamma = np.array([count ** aggressive for count in counts])
            p_winner = count_gamma / np.sum(count_gamma)

            for i, action in enumerate(action_list_observed):
                out[action] = p_winner[i] * n_observed / (n_observed + n_unobserved)
        
        # non-observed
        for _, action in enumerate(action_list_unobserved):
            out[action] = 1. / (n_observed + n_unobserved)

        return out
