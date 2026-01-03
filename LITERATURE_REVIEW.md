# Literature Review: Odds Ratio Thompson Sampling for Multi-Armed Bandits

## 1. Introduction

This literature review provides a comprehensive overview of the research landscape surrounding **Odds Ratio Thompson Sampling (ORTS)** for multi-armed bandit problems with binary outcomes. ORTS, introduced by Kim & Kim (2020), addresses a critical limitation of traditional Thompson Sampling methods: their sensitivity to time-varying treatment effects in online experimentation settings.

**Primary Reference:** [Kim, S., & Kim, K. (2020). Odds-Ratio Thompson Sampling to Control for Time-Varying Effect. arXiv:2003.01905](https://arxiv.org/abs/2003.01905)

---

## 2. Multi-Armed Bandit Problem

### 2.1 Fundamental Framework

The multi-armed bandit (MAB) problem is a classical sequential decision-making framework that captures the fundamental exploration-exploitation trade-off. At each time step $t$, an agent selects one of $K$ arms and receives a stochastic reward. The goal is to maximize cumulative reward, or equivalently, minimize *regret*—the difference between the reward obtained and the reward that would have been obtained by always selecting the optimal arm.

Formally, the expected regret after $T$ rounds is defined as:

$$R(T) = T \cdot \mu^* - \sum_{t=1}^{T} \mathbb{E}[\mu_{A_t}]$$

where $\mu^*$ is the expected reward of the optimal arm and $A_t$ is the arm selected at time $t$.

### 2.2 Classical Algorithms

Several fundamental algorithms have been developed for the MAB problem:

- **$\epsilon$-Greedy**: Selects the empirically best arm with probability $1-\epsilon$ and explores randomly with probability $\epsilon$
- **Upper Confidence Bound (UCB)**: Constructs confidence bounds and selects the arm with the highest upper bound (Auer et al., 2002)
- **Thompson Sampling**: Uses Bayesian posterior sampling to balance exploration and exploitation (Thompson, 1933)

---

## 3. Thompson Sampling

### 3.1 Overview and History

Thompson Sampling (TS) is one of the oldest heuristics for multi-armed bandit problems, dating back to Thompson (1933). It is a randomized algorithm based on Bayesian principles: at each round, the algorithm samples a parameter for each arm from its posterior distribution and selects the arm with the highest sampled value.

Despite its simplicity, TS has generated significant renewed interest due to its favorable empirical performance compared to UCB-based methods (Chapelle & Li, 2011).

### 3.2 Regret Bounds and Optimality

**Key Theoretical Results:**

| Bound Type | Result | Reference |
|------------|--------|-----------|
| Problem-independent | $O(\sqrt{KT \ln T})$ | [Agrawal & Goyal (2012)](https://arxiv.org/abs/1209.3353) |
| Problem-dependent | $(1+\epsilon)\sum_i \frac{\ln T}{\Delta_i} + O(K/\epsilon^2)$ | [Agrawal & Goyal (2013)](https://dl.acm.org/doi/10.1145/3088510) |
| Asymptotic optimality | Matches Lai-Robbins lower bound | Kaufmann et al. (2012) |

The question of Thompson Sampling's optimality for solving the stochastic MAB problem had been open since 1933. Researchers answered it positively for Bernoulli rewards by providing finite-time analysis matching the asymptotic Lai-Robbins lower bound.

**Key References:**
- [Near-Optimal Regret Bounds for Thompson Sampling (Journal of the ACM)](https://dl.acm.org/doi/10.1145/3088510)
- [Further Optimal Regret Bounds for Thompson Sampling](https://arxiv.org/abs/1209.3353)
- [A Tutorial on Thompson Sampling](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf) - Russo et al., Foundations and Trends in Machine Learning (2018)

### 3.3 Prior Sensitivity

Thompson Sampling's optimality often depends on the choice of priors. For normal distributions with unknown means and variances, TS with uniform prior achieves theoretical bounds, but TS with Jeffreys prior or reference prior cannot achieve the same bounds. This prior sensitivity becomes particularly important when models contain multiple parameters.

---

## 4. Logistic and Generalized Linear Bandits

### 4.1 Generalized Linear Bandit Framework

The generalized linear bandit (GLB) framework extends linear bandits by incorporating non-linear link functions, enabling modeling of diverse reward distributions including Bernoulli and Poisson. This framework is particularly relevant for:

- **Click prediction** in online advertising (binary click/no-click)
- **Recommendation systems** (user engagement modeling)
- **Personalized medicine** (treatment response)

The logistic bandit is a special case where rewards follow a Bernoulli distribution with probability given by the logistic (sigmoid) function:

$$P(Y=1|x) = \frac{1}{1 + e^{-\theta^T x}}$$

### 4.2 Regret Analysis for Logistic Bandits

For logistic bandits, regret guarantees of existing algorithms are $\tilde{O}(\kappa\sqrt{T})$, where $\kappa$ is a problem-dependent constant that can be arbitrarily large (scaling exponentially with the decision set size). This has motivated development of improved algorithms.

**Recent Advances:**
- [Improved Optimistic Algorithms for Logistic Bandits](https://arxiv.org/abs/2002.07530) - Faury et al. (2020)
- [Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](https://arxiv.org/abs/2507.11847) - Jointly efficient algorithms with O(1) time/space complexity per round
- [Neural Logistic Bandits](https://arxiv.org/html/2505.02069) - Combining neural networks with logistic bandit framework

### 4.3 Thompson Sampling for Logistic Bandits

Several Thompson Sampling variants have been developed for logistic/GLB settings:

- **PG-TS (Pólya-Gamma Thompson Sampling)**: Uses Pólya-Gamma augmentation for tractable posterior updates (Dumitrascu et al., NeurIPS 2018)
- [Thompson Sampling for Multinomial Logit Contextual Bandits](https://papers.nips.cc/paper/8578-thompson-sampling-for-multinomial-logit-contextual-bandits) - Achieves $\tilde{O}(d\sqrt{T})$ Bayesian regret

---

## 5. Contextual Bandits

### 5.1 Framework

Contextual bandits extend the MAB setting by incorporating side information (context) available at each decision point. The expected reward depends on both the arm selected and the observed context.

### 5.2 Key Algorithms

- **LinUCB**: Linear upper confidence bound with context features (Li et al., 2010)
- **LinTS**: Thompson Sampling with linear reward models
- [Thompson Sampling for Contextual Bandits with Linear Payoffs](https://proceedings.mlr.press/v28/agrawal13.pdf) - Agrawal & Goyal (2013)

### 5.3 Recent Developments

- [Feel-Good Thompson Sampling for Contextual Bandits](https://arxiv.org/abs/2110.00871) - Zhang (2022) introduced a modified likelihood with "feel-good bonus" for more aggressive exploration in high-dimensional settings
- [Scalable and Interpretable Contextual Bandits: A Literature Review](https://arxiv.org/abs/2505.16918) - Comprehensive 2025 survey with retail applications
- [Doubly-Adaptive Thompson Sampling](https://www.researchgate.net/publication/349682478_Doubly-Adaptive_Thompson_Sampling_for_Multi-Armed_and_Contextual_Bandits) - Adaptive exploration strategies

---

## 6. Non-Stationary and Time-Varying Bandits

### 6.1 The Challenge

In real-world applications, reward distributions often change over time. Traditional bandit algorithms assume stationarity, leading to suboptimal performance when:
- User preferences evolve
- External factors (seasonality, trends) affect outcomes
- System dynamics change

This is the core problem addressed by ORTS.

### 6.2 Approaches to Non-Stationarity

**Passive Forgetting Methods:**
- **Sliding Window**: Only uses recent observations (Garivier & Moulines, 2011)
- **Discounted UCB/TS**: Applies exponential discounting to older observations
- **f-Discounted-Sliding-Window Thompson Sampling (f-dsw TS)**: Combines discount factor with sliding windows

**Active Detection Methods:**
- **Change Point Detection**: Restarts learning when distribution changes are detected
- **Adapt-EvE**: Uses Page-Hinkley statistics for abrupt change detection

### 6.3 Recent Research

- [Non Stationary Bandits with Periodic Variation (AAMAS 2024)](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p2177.pdf) - Extensions for periodic reward distributions
- [Finite-time Analysis of Globally Nonstationary Multi-Armed Bandits (JMLR 2024)](https://jmlr.org/papers/volume25/21-0916/21-0916.pdf) - Theoretical analysis for non-stationary settings
- [LLM-Informed Multi-Armed Bandit Strategies for Non-Stationary Environments](https://www.mdpi.com/2079-9292/12/13/2814) - Using LLMs to handle distribution shifts

---

## 7. Industry Applications

### 7.1 A/B Testing vs. Multi-Armed Bandits

Traditional A/B testing splits traffic equally between variants, which can be inefficient when one variant is clearly superior. Multi-armed bandits enable:
- **Dynamic traffic allocation** to better-performing variants
- **Faster convergence** to optimal decisions
- **Reduced opportunity cost** of testing

However, MABs assume consistent conversion rates throughout testing, making them sensitive to time-varying effects—motivating the need for methods like ORTS.

**References:**
- [A Practical Guide to Multi Armed Bandit A/B Testing](https://splitmetrics.com/blog/multi-armed-bandit-in-a-b-testing/)
- [Multi-Armed Bandits vs. A/B Testing: Choosing the Right Approach](https://amplitude.com/blog/multi-armed-bandit-vs-ab-testing)

### 7.2 Case Studies

**Netflix:**
Netflix employs contextual bandits for personalized artwork selection, dynamically choosing thumbnail images to maximize user engagement. The system handles delayed rewards across multiple user sessions and precomputes personalized selections for scale.

**Spotify:**
In March 2025, [Spotify deployed a contextual bandit system](https://research.atspotify.com/2025/9/calibrated-recommendations-with-contextual-bandits-on-spotify-homepage) for calibrating content-type distributions on the Home page, balancing music, podcasts, and audiobooks based on evolving user preferences.

**Lyft:**
Lyft uses contextual bandits for the "Wait and Save" feature, dynamically setting pickup windows based on local driver availability and demand context.

**References:**
- [How Netflix, Lyft, and Yahoo use Contextual Bandits for Personalization](https://www.geteppo.com/blog/netflix-lyft-yahoo-contextual-bandits)
- [Carousel Personalization in Music Streaming Apps with Contextual Bandits](https://arxiv.org/pdf/2009.06546) (Deezer research)

---

## 8. Odds Ratio Thompson Sampling (ORTS)

### 8.1 Motivation

Traditional Thompson Sampling methods for logistic bandits maintain posteriors over absolute arm parameters. When time-varying effects (e.g., seasonal trends, platform changes) affect all arms simultaneously, these methods can become unreliable because:

1. Absolute conversion rates change, invalidating historical estimates
2. The posterior becomes misspecified relative to current conditions
3. Exploration-exploitation balance is disrupted

### 8.2 Key Innovation

ORTS reparameterizes the logistic model using **odds ratios** between arms rather than absolute parameters. The key insight is that while absolute conversion rates may drift over time, the *relative* performance between arms often remains more stable.

**Odds Ratio Parameterization:**
Instead of modeling $P(Y=1|arm=k)$ directly, ORTS models:
$$OR_{k,j} = \frac{P(Y=1|arm=k)/(1-P(Y=1|arm=k))}{P(Y=1|arm=j)/(1-P(Y=1|arm=j))}$$

### 8.3 Advantages

1. **Robustness to Time-Varying Effects**: Background changes affecting all arms equally cancel out in odds ratios
2. **Efficient Parameter Space**: When arms are added/removed, only relevant odds ratios need updating
3. **Minimal Performance Degradation**: In stable environments, ORTS performs comparably to full-rank methods
4. **Practical Applicability**: Validated on real-world service data

### 8.4 Algorithm Components

The ORTS implementation includes:
- **Bayesian posterior estimation** using MAP with Gaussian priors
- **Monte Carlo sampling** for computing winning probabilities
- **Precision matrix estimation** via Fisher information
- **Discount parameter** for additional temporal adaptation

---

## 9. Related Methods and Comparisons

### 9.1 Full-Rank Thompson Sampling (Full-TS)

Maintains full posterior on all arm-specific parameters. More standard Bayesian approach but less robust when the environment changes.

### 9.2 Beta-Bernoulli Thompson Sampling

Uses conjugate Beta priors for independent Bernoulli arms. Simple and computationally efficient but:
- Ignores correlations between arms
- Sensitive to time-varying effects
- Cannot leverage shared information across arms

### 9.3 Comparison Summary

| Method | Robustness to Time-Varying Effects | Computational Efficiency | Arm Correlations |
|--------|-----------------------------------|-------------------------|------------------|
| ORTS | High | Moderate | Modeled via odds ratios |
| Full-TS | Low | Moderate | Fully modeled |
| Beta-Bernoulli TS | Low | High | Independent |

---

## 10. Future Directions

### 10.1 Theoretical Extensions

- **Regret bounds for ORTS**: Formal analysis of regret under time-varying conditions
- **Optimal discount rates**: Adaptive methods for setting the discount parameter
- **Extensions to contextual settings**: ORTS with context features

### 10.2 Algorithmic Improvements

- **Online mirror descent integration**: Following recent GLB advances for computational efficiency
- **Neural network parameterization**: Combining ORTS principles with deep learning
- **Hybrid approaches**: Detecting when to use ORTS vs. full-rank methods

### 10.3 Applications

- **Sequential experimentation platforms**: Integration with industrial A/B testing systems
- **Recommendation systems**: Handling evolving user preferences
- **Dynamic pricing**: Adapting to market condition changes

---

## 11. Conclusion

Odds Ratio Thompson Sampling represents an important advancement in multi-armed bandit methodology, addressing the practical challenge of time-varying treatment effects that plagues traditional approaches. By reparameterizing the problem in terms of odds ratios, ORTS achieves robustness to background effects while maintaining competitive performance in stable environments.

The method bridges the gap between theoretical Thompson Sampling research and practical requirements of real-world online experimentation, where environmental conditions shift over time and new variants are frequently added or removed. As demonstrated by industry applications at Netflix, Spotify, and other major platforms, bandit methods are increasingly central to personalization and optimization—making robust approaches like ORTS particularly valuable.

---

## References

### Core Paper
- Kim, S., & Kim, K. (2020). [Odds-Ratio Thompson Sampling to Control for Time-Varying Effect](https://arxiv.org/abs/2003.01905). arXiv:2003.01905.

### Thompson Sampling Foundations
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.
- Chapelle, O., & Li, L. (2011). An empirical evaluation of Thompson sampling. NIPS.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on Thompson sampling. Foundations and Trends in Machine Learning, 11(1), 1-96.
- Agrawal, S., & Goyal, N. (2012). [Analysis of Thompson Sampling for the Multi-armed Bandit Problem](https://arxiv.org/abs/1111.1797). COLT.
- Agrawal, S., & Goyal, N. (2013). [Further Optimal Regret Bounds for Thompson Sampling](https://arxiv.org/abs/1209.3353). AISTATS.

### Logistic and Generalized Linear Bandits
- Filippi, S., Cappe, O., Garivier, A., & Szepesvári, C. (2010). Parametric bandits: The generalized linear case. NIPS.
- Faury, L., Abeille, M., Calauzènes, C., & Fercoq, O. (2020). [Improved Optimistic Algorithms for Logistic Bandits](https://arxiv.org/abs/2002.07530). ICML.
- Dumitrascu, B., et al. (2018). PG-TS: Improved Thompson Sampling for Logistic Contextual Bandits. NeurIPS.

### Non-Stationary Bandits
- Garivier, A., & Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. ALT.
- Besbes, O., Gur, Y., & Zeevi, A. (2014). Stochastic multi-armed-bandit problem with non-stationary rewards. NIPS.

### Contextual Bandits
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. WWW.
- Agrawal, S., & Goyal, N. (2013). [Thompson Sampling for Contextual Bandits with Linear Payoffs](https://proceedings.mlr.press/v28/agrawal13.pdf). ICML.
- Zhang, T. (2022). [Feel-Good Thompson Sampling for Contextual Bandits](https://arxiv.org/abs/2110.00871).

### Industry Applications
- [Calibrated Recommendations with Contextual Bandits on Spotify Homepage](https://research.atspotify.com/2025/9/calibrated-recommendations-with-contextual-bandits-on-spotify-homepage). Spotify Research, 2025.
- [How Netflix, Lyft, and Yahoo use Contextual Bandits](https://www.geteppo.com/blog/netflix-lyft-yahoo-contextual-bandits). Eppo Blog.

### Surveys and Tutorials
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.
- [Scalable and Interpretable Contextual Bandits: A Literature Review](https://arxiv.org/abs/2505.16918). arXiv, 2025.
