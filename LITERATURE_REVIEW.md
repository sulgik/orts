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


# Literature Review: Odds-Ratio Thompson Sampling for Multi-Armed Bandits

**Author:** Literature review for the ORTS (Odds-Ratio Thompson Sampling) project
**Date:** January 2026
**Repository:** [sulgik/orts](https://github.com/sulgik/orts)

---

## 1. Introduction

Multi-armed bandit (MAB) problems represent a fundamental challenge in sequential decision-making under uncertainty. These problems involve balancing exploration (learning about uncertain options) with exploitation (choosing the seemingly best option based on current knowledge). This literature review surveys the relevant research areas that underpin the Odds-Ratio Thompson Sampling (ORTS) algorithm, with particular focus on handling time-varying effects in bandit problems.

---

## 2. Multi-Armed Bandit Problems

### 2.1 Classical Framework

The multi-armed bandit problem, first formalized in the early 20th century, models a scenario where a decision-maker must repeatedly choose among multiple alternatives (arms), each yielding stochastic rewards. The objective is to maximize cumulative reward over time while learning which arms perform best.

The performance of bandit algorithms is typically measured by **regret**: the difference between the cumulative reward obtained by an optimal policy (which always selects the best arm) and the reward obtained by the learning algorithm.

### 2.2 Key Algorithms

Traditional approaches to the MAB problem include:

- **ε-Greedy**: Exploits the currently best-known arm with probability 1-ε and explores randomly with probability ε
- **Upper Confidence Bound (UCB)**: Selects arms based on optimistic estimates that balance mean rewards with uncertainty
- **Thompson Sampling**: Uses Bayesian posterior sampling to naturally balance exploration and exploitation

### 2.3 Non-Stationary Bandits

Real-world applications often violate the stationarity assumption of classical bandits. Several recent works address this challenge:

#### Slowly-Varying Non-Stationary Bandits (2025)

Recent research focuses on settings where arms can change arbitrarily over time as long as the amount of change in their mean rewards between successive time steps is bounded uniformly across the horizon. This framework applies to:
- Slowly drifting distributions in natural language tasks
- Data from physical transducers
- Slowly fading wireless channels

**Source:** [On Slowly-Varying Non-Stationary Bandits](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_303.pdf), RLJ Conference 2025

#### Bridging Adversarial and Nonstationary Bandits (2025)

Chen, Yang, and Zhang (2025) provide a unified formulation that bridges adversarial and nonstationary bandit formulations as special cases, with algorithms that attain optimal regret bounds. This work demonstrates that different modeling approaches for time-varying rewards can be understood within a common framework.

**Source:** [Bridging Adversarial and Nonstationary Multi-Armed Bandit](https://journals.sagepub.com/doi/abs/10.1177/10591478251313780)

#### Total Variation Budgeted Bandits

The "drifting" or "continuously changing" bandits setting constrains the total amount of successive changes in arms' reward means, providing theoretical guarantees when change is bounded.

**Source:** [Nonstationary Stochastic Multiarmed Bandits](https://arxiv.org/abs/2101.08980)


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


### 3.1 Historical Background and Theory

Thompson Sampling, originally introduced by William R. Thompson in 1933, represents a Bayesian approach to the exploration-exploitation tradeoff. The algorithm maintains a probability distribution over the parameters of each arm's reward distribution and selects arms by sampling from these posterior distributions.

**Key theoretical result:** Agrawal and Goyal (2012) proved that Thompson Sampling achieves logarithmic expected regret for the stochastic multi-armed bandit problem, matching the asymptotic lower bound.

**Source:** [Analysis of Thompson Sampling for the Multi-armed Bandit Problem](https://arxiv.org/abs/1111.1797)

### 3.2 The Algorithm

The basic Thompson Sampling procedure:

1. Initialize prior distributions for each arm's reward parameters
2. At each time step:
   - Sample parameters from each arm's posterior distribution
   - Select the arm with the highest sampled expected reward
   - Observe the reward and update the posterior for the selected arm

This approach naturally balances exploration and exploitation: arms with uncertain posteriors have higher variance in their samples, increasing their probability of being selected.

### 3.3 Applications

Thompson Sampling has been widely adopted in:
- Online advertising and A/B testing
- Clinical trials
- Recommendation systems
- Website optimization
- Decentralized decision-making

**Sources:**
- [Thompson Sampling: A Powerful Algorithm for Multi-Armed Bandit Problems](https://medium.com/@iqra.bismi/thompson-sampling-a-powerful-algorithm-for-multi-armed-bandit-problems-95c15f63a180)
- [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling)

---

## 4. Contextual Bandits and Logistic Regression

### 4.1 Contextual Bandit Framework

Contextual bandits extend the basic MAB problem by incorporating context (side information) available at each decision point. The goal is to learn a mapping from contexts to optimal actions, enabling personalization and adaptive decision-making.

### 4.2 Logistic Thompson Sampling

When rewards are binary (e.g., click/no-click, success/failure), logistic regression provides a natural model for contextual bandits. However, combining logistic regression with Thompson Sampling presents challenges.

#### PG-TS: Improved Thompson Sampling for Logistic Contextual Bandits (NeurIPS 2018)

This foundational work addresses a key challenge: the logistic regression likelihood leads to an **intractable posterior** — the necessary integrals are not available in closed form and are difficult to approximate. This intractability makes the sampling step of Thompson Sampling with binary or categorical rewards challenging.

The PG-TS algorithm relies on Online Logistic Regression to learn an independent normal distribution for each linear model weight, enabling effective exploration-exploitation tradeoffs.

**Source:** [PG-TS: Improved Thompson Sampling for Logistic Contextual Bandits](https://proceedings.neurips.cc/paper_files/paper/2018/file/ce6c92303f38d297e263c7180f03d402-Paper.pdf)

#### Recent Advances (2024-2025)

**Feel-Good Thompson Sampling for Contextual Bandits (NeurIPS 2025):**
This recent work presents the first systematic evaluation of Feel-Good Thompson Sampling (FGTS) and its variants under exact and approximate MCMC-derived posterior regimes. The study found that high-fidelity posteriors combined with FGTS's optimism reduce regret in linear and logistic settings.

**Source:** [Feel-Good Thompson Sampling for Contextual Bandits](https://arxiv.org/html/2507.15290)

**Scalable and Interpretable Contextual Bandits (May 2025):**
This work leverages logistic regression with static features in stationary environments, exposing model weights for improved interpretability. The paper recommends comparative analysis with Logistic Thompson Sampling for performance evaluation.

**Source:** [Scalable and Interpretable Contextual Bandits](https://arxiv.org/html/2505.16918v1)

### 4.3 Other Approaches

**Thompson Sampling for Multinomial Logit:**
When choices involve multiple discrete options simultaneously, multinomial logit models extend binary logistic regression. Thompson Sampling has been adapted to this setting for product assortment and choice modeling.

**Source:** [Thompson Sampling for Multinomial Logit Contextual Bandits](https://papers.nips.cc/paper/8578-thompson-sampling-for-multinomial-logit-contextual-bandits)

---

## 5. Odds Ratios in Statistical Modeling

### 5.1 Definition and Interpretation

The **odds** of an event are defined as the probability that the event occurs divided by the probability that it does not occur:

$$\text{Odds} = \frac{P(\text{event})}{1 - P(\text{event})}$$

For example, if the probability of mortality is 0.3, the odds of dying are 0.3/(1-0.3) = 0.43.

The **odds ratio (OR)** compares the odds of an event occurring in two different conditions or groups:

$$\text{OR} = \frac{\text{Odds}_1}{\text{Odds}_2}$$

### 5.2 Odds Ratios in Logistic Regression

In logistic regression, coefficients represent log-odds:

$$\log\left(\frac{P(Y=1|X)}{1-P(Y=1|X)}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots$$

The odds ratio is obtained by exponentiating the regression coefficient: **OR = exp(β)**

**Interpretation:**
- OR > 1: The event is more likely as the predictor increases
- OR < 1: The event is less likely as the predictor increases
- OR = 1: No association between predictor and outcome

### 5.3 Advantages of Odds Ratio Parameterization

1. **Invariance to reference group:** Odds ratios comparing two treatments remain the same regardless of which additional treatments are included in the model
2. **Multiplicative interpretation:** Effects combine multiplicatively on the odds scale
3. **Robustness:** Less sensitive to baseline probability changes than raw probabilities
4. **Clinical/practical interpretation:** Well-established in epidemiology and medical research

**Sources:**
- [FAQ: How do I interpret odds ratios in logistic regression?](https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/)
- [Understanding logistic regression analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3936971/)
- [Explaining Odds Ratios](https://pmc.ncbi.nlm.nih.gov/articles/PMC2938757/)

---

## 6. Odds-Ratio Thompson Sampling (ORTS)

### 6.1 The Core Paper

**Title:** Odds-Ratio Thompson Sampling to Control for Time-Varying Effect

**Authors:** Sulgi Kim and Kyungmin Kim

**Publication:** [arXiv:2003.01905](https://arxiv.org/abs/2003.01905)

### 6.2 Motivation and Problem Statement

Continuous online experiments face a critical challenge: **temporal background effects** that can confound experimental results. For example:
- **Day-of-week effects:** User behavior varies systematically between weekdays and weekends
- **Seasonal trends:** Purchase patterns change during holidays or specific seasons
- **External events:** News events, competitor actions, or marketing campaigns affect user behavior

Standard Thompson Sampling with full logistic regression can be **overly sensitive** to these time-varying effects because it updates all parameters, including baseline effects that may fluctuate for reasons unrelated to treatment efficacy.

### 6.3 Key Innovation: Odds Ratio Reparameterization

ORTS reparameterizes the logistic model using **odds ratio parameters**, enabling Thompson Sampling to work with a **subset of parameters** that capture relative treatment effects while being invariant to baseline shifts.

**Mathematical Framework:**

Instead of modeling absolute log-odds:
$$\log(\text{odds}_i) = \alpha_i$$

ORTS models odds ratios relative to a reference arm:
$$\log(\text{OR}_{i,\text{ref}}) = \alpha_i - \alpha_{\text{ref}}$$

This parameterization has a crucial property: **odds ratios remain stable even when baseline probabilities shift due to time-varying effects**.

### 6.4 The ORTS Algorithm

1. **Prior specification:** Maintain Bayesian priors only on odds ratio parameters (not on baseline)
2. **Posterior updates:** Update only odds ratio parameters when new data arrives, using odds-ratio-only updates
3. **Sampling:** Sample from the odds ratio posterior distributions
4. **Arm selection:** Select the arm with the highest sampled odds ratio
5. **Fallback to full-rank:** Optionally perform full-rank updates periodically for efficiency

### 6.5 Theoretical Properties

**Robustness to Time-Varying Effects:**
By focusing inference on odds ratios rather than absolute probabilities, ORTS maintains stable estimates of relative treatment effects even when:
- Baseline conversion rates drift over time
- Periodic patterns affect all arms similarly
- External shocks impact overall user behavior

**Trade-off:**
The algorithm accepts marginal performance loss in stable environments in exchange for robustness in dynamic settings — a favorable trade-off for production systems.

### 6.6 Empirical Validation

**Simulation Studies:**
The authors demonstrate that ORTS maintains robust performance against temporal background effects with minimal regret increase compared to optimal policies.

**Real-World Service Data:**
Testing on actual production data shows ORTS generates greater cumulative rewards compared to full-rank Thompson Sampling when time-varying effects are present.

### 6.7 Practical Implementation

The [orts repository](https://github.com/sulgik/orts) provides a Python implementation with key features:

- **Flexible updating:** Support for both odds-ratio-only and full-rank updates
- **Discount factor:** Optional exponential discounting for non-stationary environments
- **Dynamic arm management:** Handle arms entering and leaving the experiment
- **Win probability computation:** Monte Carlo estimation of arm selection probabilities

**Example usage:**
```python
from logisticbandit import LogisticBandit

orpar = LogisticBandit()

# Odds-ratio-only update (default, robust to time-varying effects)
obs = {"arm_1": [30000, 300], "arm_2": [30000, 290]}
orpar.update(obs)

# Get win probabilities for arm selection
orpar.win_prop()

# Optional: full-rank update for efficiency
obs = {"arm_1": [30000, 310], "arm_3": [30000, 300]}
orpar.update(obs, odds_ratios_only=False)
```

---

## 7. Related Work and Positioning

### 7.1 Comparison with Discounting Methods

Traditional approaches to non-stationary bandits often use **exponential discounting** or **sliding windows** to down-weight old observations. ORTS differs fundamentally:

- **Discounting:** Assumes all parameters are time-varying, discards historical information
- **ORTS:** Distinguishes between stable relative effects and time-varying baseline effects, retaining relevant historical information

### 7.2 Comparison with Contextual Bandits

Contextual bandits with time features can model temporal effects explicitly. However:

- **Requires knowing the temporal structure:** Must specify day-of-week, seasonal patterns, etc.
- **ORTS is agnostic:** Works without modeling the specific form of time-varying effects
- **Complementary approaches:** ORTS could potentially be combined with contextual features

### 7.3 Relationship to Difference-in-Differences

The odds ratio parameterization shares conceptual similarities with **difference-in-differences** estimation in econometrics, which focuses on relative changes while controlling for baseline trends. Both approaches recognize that relative comparisons can be more stable than absolute measurements.

---

## 8. Open Questions and Future Directions

### 8.1 Theoretical Extensions

1. **Regret bounds:** Formal regret analysis for ORTS under different time-varying models
2. **Optimal update schedules:** When to use odds-ratio-only vs. full-rank updates
3. **Multiple reference arms:** Extensions to settings with multiple stable reference groups

### 8.2 Algorithmic Improvements

1. **Adaptive switching:** Automatically detect when full-rank updates are beneficial
2. **Hierarchical models:** Incorporate structure across arms (e.g., similar treatments)
3. **Non-logistic models:** Extend odds-ratio principles to other generalized linear models

### 8.3 Application Domains

1. **Healthcare trials:** Clinical trials with seasonal effects or evolving standard-of-care
2. **E-commerce:** Product recommendations with time-varying user preferences
3. **Content optimization:** A/B testing for content platforms with trending topics
4. **Advertising:** Campaign optimization under varying market conditions

### 8.4 Integration with Modern ML

1. **Neural bandits:** Combining odds-ratio robustness with deep learning flexibility
2. **Causal inference:** Stronger connections to causal effect estimation under temporal confounding
3. **Multi-objective optimization:** Handling time-varying constraints and multiple KPIs

---

## 9. Conclusion

The Odds-Ratio Thompson Sampling algorithm represents an important contribution to practical bandit algorithms for real-world deployment. By recognizing that relative treatment effects are often more stable than absolute probabilities, ORTS provides a principled approach to robust experimentation under time-varying conditions.

Key insights from the literature:

1. **Thompson Sampling** provides theoretically grounded exploration-exploitation balance
2. **Logistic regression** enables modeling of binary rewards but creates posterior sampling challenges
3. **Non-stationary bandits** are essential for real-world applications but require specialized algorithms
4. **Odds ratios** offer invariance properties that make them well-suited for time-varying environments
5. **ORTS** combines these ideas into a practical algorithm with demonstrated empirical success

As online experimentation continues to grow in importance across industries, algorithms like ORTS that balance statistical rigor with practical robustness will become increasingly valuable.

---

## 10. References

### Primary Source
- Kim, S., & Kim, K. (2020). Odds-Ratio Thompson Sampling to Control for Time-Varying Effect. arXiv:2003.01905. https://arxiv.org/abs/2003.01905

### Multi-Armed Bandits - Foundations
- Agrawal, S., & Goyal, N. (2012). Analysis of Thompson Sampling for the Multi-armed Bandit Problem. PMLR. https://arxiv.org/abs/1111.1797
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.

### Non-Stationary Bandits (2024-2025)
- Krishnamurthy, A., & Gopalan, A. (2025). On Slowly-Varying Non-Stationary Bandits. RLJ Conference 2025. https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_303.pdf
- Chen, N., Yang, S., & Zhang, H. (2025). Bridging Adversarial and Nonstationary Multi-Armed Bandit. SAGE Journals. https://journals.sagepub.com/doi/abs/10.1177/10591478251313780
- Besbes, O., Gur, Y., & Zeevi, A. (2021). Nonstationary Stochastic Multiarmed Bandits: UCB Policies and Minimax Regret. arXiv:2101.08980. https://arxiv.org/abs/2101.08980

### Logistic Thompson Sampling (2024-2025)
- Feel-Good Thompson Sampling for Contextual Bandits (NeurIPS 2025). https://arxiv.org/html/2507.15290
- Scalable and Interpretable Contextual Bandits: A Literature Review (2025). https://arxiv.org/html/2505.16918v1
- Kveton, B., et al. (2018). PG-TS: Improved Thompson Sampling for Logistic Contextual Bandits. NeurIPS 2018. https://proceedings.neurips.cc/paper_files/paper/2018/file/ce6c92303f38d297e263c7180f03d402-Paper.pdf

### Contextual Bandits
- Agrawal, S., & Goyal, N. (2013). Thompson sampling for contextual bandits with linear payoffs. ICML 2013.
- Oh, M. H., & Iyengar, G. (2019). Thompson Sampling for Multinomial Logit Contextual Bandits. NeurIPS 2019. https://papers.nips.cc/paper/8578-thompson-sampling-for-multinomial-logit-contextual-bandits

### Odds Ratios in Statistics
- UCLA Statistical Consulting. FAQ: How do I interpret odds ratios in logistic regression? https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
- Park, H. A. (2013). Understanding logistic regression analysis. Epidemiology and Health, 35, e2013003. https://pmc.ncbi.nlm.nih.gov/articles/PMC3936971/
- Sinclair, J. C., & Bracken, M. B. (1994). Explaining Odds Ratios. Journal of Clinical Epidemiology, 47(4), 355-356. https://pmc.ncbi.nlm.nih.gov/articles/PMC2938757/

### General Resources
- Thompson Sampling - Wikipedia. https://en.wikipedia.org/wiki/Thompson_sampling
- Multi-armed bandit - Wikipedia. https://en.wikipedia.org/wiki/Multi-armed_bandit
- Medium: Thompson Sampling: A Powerful Algorithm for Multi-Armed Bandit Problems. https://medium.com/@iqra.bismi/thompson-sampling-a-powerful-algorithm-for-multi-armed-bandit-problems-95c15f63a180
- Towards Data Science: Thompson Sampling. https://towardsdatascience.com/thompson-sampling-fc28817eacb8/

---

**Document Version:** 1.0
**Last Updated:** January 3, 2026
**Maintained by:** ORTS Project Contributors
