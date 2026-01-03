# Literature Review: Odds Ratio Thompson Sampling

## 1. 서론

본 문서는 **Odds Ratio Thompson Sampling (ORTS)**의 연구 배경과 관련 문헌을 정리한다. ORTS는 Kim & Kim (2020)이 제안한 방법으로, 시간에 따라 변하는 처리 효과(time-varying treatment effect)에 강건한 Multi-Armed Bandit 알고리즘이다.

> **핵심 논문:** Kim, S., & Kim, K. (2020). [Odds-Ratio Thompson Sampling to Control for Time-Varying Effect](https://arxiv.org/abs/2003.01905). arXiv:2003.01905.

---

## 2. Multi-Armed Bandit 문제

### 2.1 기본 프레임워크

Multi-Armed Bandit (MAB)은 탐색(exploration)과 활용(exploitation) 간의 균형을 다루는 순차적 의사결정 문제이다. 매 시점 $t$에서 에이전트는 $K$개의 arm 중 하나를 선택하고 확률적 보상을 받는다.

**Regret 정의:**
$$R(T) = T \cdot \mu^* - \sum_{t=1}^{T} \mathbb{E}[\mu_{A_t}]$$

- $\mu^*$: 최적 arm의 기대 보상
- $A_t$: 시점 $t$에서 선택한 arm

### 2.2 주요 알고리즘

| 알고리즘 | 핵심 아이디어 |
|---------|-------------|
| ε-Greedy | 확률 ε로 랜덤 탐색, 1-ε로 최적 arm 선택 |
| UCB | 신뢰구간 상한이 가장 높은 arm 선택 |
| Thompson Sampling | 사후분포에서 샘플링하여 arm 선택 |

---

## 3. Thompson Sampling

### 3.1 개요

Thompson Sampling (TS)은 1933년 Thompson이 제안한 베이지안 기반 알고리즘이다. 매 라운드마다 각 arm의 사후분포에서 파라미터를 샘플링하고, 가장 높은 값을 가진 arm을 선택한다.

**알고리즘:**
1. 각 arm $k$에 대해 사후분포 $\pi_k(\theta)$에서 $\hat{\theta}_k$ 샘플링
2. $\arg\max_k \hat{\theta}_k$ 선택
3. 보상 관측 후 사후분포 업데이트

### 3.2 Regret Bounds

| 유형 | Bound | 출처 |
|-----|-------|-----|
| Problem-independent | $O(\sqrt{KT \ln T})$ | Agrawal & Goyal (2012) |
| Problem-dependent | $(1+\epsilon)\sum_i \frac{\ln T}{\Delta_i}$ | Agrawal & Goyal (2013) |
| 점근적 최적성 | Lai-Robbins 하한과 일치 | Kaufmann et al. (2012) |

### 3.3 핵심 문헌

- Russo et al. (2018). [A Tutorial on Thompson Sampling](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf). Foundations and Trends in ML.
- Chapelle & Li (2011). An Empirical Evaluation of Thompson Sampling. NIPS.

---

## 4. Logistic Bandit

### 4.1 Generalized Linear Bandit

Generalized Linear Bandit (GLB)은 보상이 비선형 link 함수를 통해 결정되는 확장된 프레임워크이다. Logistic bandit은 이진 보상(클릭/비클릭)을 모델링하는 중요한 특수 케이스이다.

**모델:**
$$P(Y=1|x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

### 4.2 응용 분야

- **온라인 광고**: 클릭률(CTR) 최적화
- **추천 시스템**: 사용자 참여도 예측
- **A/B 테스팅**: 전환율 최적화

### 4.3 관련 연구

- Filippi et al. (2010). Parametric Bandits: The Generalized Linear Case. NIPS.
- Faury et al. (2020). [Improved Optimistic Algorithms for Logistic Bandits](https://arxiv.org/abs/2002.07530). ICML.
- Dumitrascu et al. (2018). PG-TS: Improved Thompson Sampling for Logistic Contextual Bandits. NeurIPS.

---

## 5. Non-Stationary Bandit

### 5.1 문제 정의

실제 환경에서는 보상 분포가 시간에 따라 변할 수 있다:
- 사용자 선호도 변화
- 계절적 요인
- 외부 환경 변화

기존 MAB 알고리즘은 정상성(stationarity)을 가정하므로 이러한 환경에서 성능이 저하된다.

### 5.2 기존 접근법

**Passive 방법:**
- **Sliding Window**: 최근 관측만 사용 (Garivier & Moulines, 2011)
- **Discounting**: 과거 관측에 가중치 감소

**Active 방법:**
- **Change Point Detection**: 분포 변화 감지 시 재시작
- **Adapt-EvE**: Page-Hinkley 통계량 활용

### 5.3 한계

기존 방법들은:
1. 변화 시점을 정확히 감지하기 어려움
2. 모든 arm의 정보를 리셋해야 함
3. 상대적 성능 차이는 유지될 수 있음을 활용하지 못함

---

## 6. ORTS: Odds Ratio Thompson Sampling

### 6.1 핵심 아이디어

ORTS의 핵심 통찰: **절대적 전환율은 변해도 arm 간 상대적 성능(odds ratio)은 안정적으로 유지된다.**

**Odds Ratio 정의:**
$$OR_{k,j} = \frac{p_k / (1-p_k)}{p_j / (1-p_j)}$$

시간에 따른 배경 효과가 모든 arm에 동일하게 적용되면, odds ratio는 일정하게 유지된다.

### 6.2 방법론

**재매개변수화:**
- 기존: 각 arm의 절대 파라미터 $\beta_k$ 추정
- ORTS: 기준 arm 대비 odds ratio $\beta_k - \beta_0$ 추정

**장점:**
1. 배경 효과에 강건
2. Arm 추가/제거 시 효율적
3. 안정적 환경에서도 성능 저하 없음

### 6.3 알고리즘 구성요소

```
LogisticBandit 클래스:
├── update(): 베이지안 사후분포 업데이트 (MAP + Fisher Information)
├── win_prop(): Monte Carlo 샘플링으로 승률 계산
├── get_par(): 특정 arm subset의 파라미터 조회
└── transform(): 기준 arm 변경
```

---

## 7. 방법 비교

| 방법 | Time-Varying 강건성 | 계산 효율성 | Arm 상관관계 |
|-----|-------------------|-----------|------------|
| **ORTS** | 높음 | 중간 | Odds ratio로 모델링 |
| Full-TS | 낮음 | 중간 | 전체 공분산 |
| Beta-Bernoulli TS | 낮음 | 높음 | 독립 가정 |

---

## 8. 산업 적용 사례

### 8.1 A/B 테스팅

전통적 A/B 테스팅은 트래픽을 균등 분배하지만, MAB는 성능 좋은 variant에 더 많은 트래픽을 동적 할당한다. 그러나 시간에 따른 효과 변화에 취약하여 ORTS 같은 방법이 필요하다.

### 8.2 사례

| 회사 | 적용 | 참고 |
|-----|-----|-----|
| Netflix | 개인화 썸네일 선택 | Contextual Bandit |
| Spotify | 홈페이지 콘텐츠 배치 | [Spotify Research (2025)](https://research.atspotify.com/) |
| Lyft | "Wait and Save" 픽업 시간 | Contextual Bandit |

---

## 9. 향후 연구 방향

1. **이론적 분석**: Non-stationary 환경에서 ORTS의 regret bound 도출
2. **Contextual 확장**: Context 정보를 활용한 ORTS
3. **적응적 discount**: 환경 변화에 따른 자동 discount rate 조정
4. **Deep Learning 결합**: Neural network 기반 odds ratio 추정

---

## 참고문헌

### 핵심 논문
- Kim, S., & Kim, K. (2020). Odds-Ratio Thompson Sampling to Control for Time-Varying Effect. arXiv:2003.01905.

### Thompson Sampling
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another. Biometrika.
- Agrawal, S., & Goyal, N. (2012). Analysis of Thompson Sampling for the Multi-armed Bandit Problem. COLT.
- Agrawal, S., & Goyal, N. (2013). Further Optimal Regret Bounds for Thompson Sampling. AISTATS.
- Russo, D., et al. (2018). A Tutorial on Thompson Sampling. Foundations and Trends in ML.

### Logistic/GLB Bandit
- Filippi, S., et al. (2010). Parametric Bandits: The Generalized Linear Case. NIPS.
- Faury, L., et al. (2020). Improved Optimistic Algorithms for Logistic Bandits. ICML.
- Li, L., et al. (2017). Provably Optimal Algorithms for Generalized Linear Contextual Bandits. ICML.

### Non-Stationary Bandit
- Garivier, A., & Moulines, E. (2011). On Upper-Confidence Bound Policies for Switching Bandit Problems. ALT.
- Besbes, O., et al. (2014). Stochastic Multi-Armed-Bandit Problem with Non-Stationary Rewards. NIPS.

### 교과서
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.
