from typing import Dict, List
import numpy as np

class TSPar(object):
    """
    Beta-Bernoulli Thompson Sampling for multi-armed bandits.

    This is a traditional Thompson Sampling approach using Beta-Bernoulli
    conjugate priors. It's simpler than LogisticBandit but may be less
    robust to time-varying effects.

    Attributes
    ----------
    action_list : list
        List of action names.
    alpha : numpy.ndarray
        Alpha parameters (success counts + 1) for Beta distributions.
    beta : numpy.ndarray
        Beta parameters (failure counts + 1) for Beta distributions.

    Examples
    --------
    >>> tspar = TSPar()
    >>> obs = {"arm_1": [1000, 50], "arm_2": [1000, 45]}
    >>> tspar.update(obs)
    >>> tspar.win_prop()
    """

    def __init__(self) -> None:
        self.action_list: List[str] = []
        self.alpha: np.ndarray = np.array([])
        self.beta: np.ndarray = np.array([])

    def get_models(self) -> List[str]:
        """
        Get the list of currently tracked actions.

        Returns
        -------
        list
            List of action names.
        """
        return self.action_list

    def update(self, obs: Dict[str, List[float]],
               alpha_0: float = 0.01, max_iter: int = 100) -> None:
        """
        Update Beta distributions with new observations.

        Parameters
        ----------
        obs : dict
            Dictionary mapping action names to [total_count, success_count].
            Example: {"arm_1": [1000, 50], "arm_2": [1000, 45]}
        alpha_0 : float, optional
            Learning rate parameter (currently unused). Default is 0.01.
        max_iter : int, optional
            Maximum iterations parameter (currently unused). Default is 100.

        Returns
        -------
        None
            Updates self.alpha and self.beta in place.

        Raises
        ------
        ValueError
            If obs is empty or observation format is invalid.
        """
        # Input validation
        if not obs:
            raise ValueError("obs dictionary cannot be empty")

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

        if len(self.action_list) == 0:
            action_list = [i for i in obs.keys()]
            self.action_list = action_list
            self.alpha = np.array([1]*len(action_list))
            self.beta  = np.array([1]*len(action_list))

        views  = np.array([obs[action][0] for action in self.action_list])
        clicks = np.array([obs[action][1] for action in self.action_list])

        self.alpha = self.alpha + clicks
        self.beta  = self.beta + views - clicks

    def win_prop(self, draw: int = 10000) -> Dict[str, float]:
        """
        Calculate winning probabilities using Thompson Sampling.

        Parameters
        ----------
        draw : int, optional
            Number of Monte Carlo samples. Default is 10000.

        Returns
        -------
        dict
            Dictionary mapping action names to winning probabilities.

        Raises
        ------
        ValueError
            If draw is not positive.
        """
        # Input validation
        if draw <= 0:
            raise ValueError("draw must be positive, got {}".format(draw))

        mc = np.matrix(np.random.beta(self.alpha, self.beta, size=[draw, len(self.action_list)]))

        # count frequency of each arm being winner 
        counts = [0.0 for _ in range(len(self.action_list))]
        winner_idxs = np.asarray(mc.argmax(axis = 1)).reshape(draw, )
        for idx in winner_idxs:
            counts[idx] += 1.0
        
        # divide by draw to approximate probability distribution
        p_winner = [count / float(draw) for count in counts]

        out = {self.action_list[i]: p_winner[i] for i in range(len(p_winner))}
        return out
    
