# Tutorial: Multi-Armed Bandit과 Thompson Sampling 완벽 가이드

## 목차

1. [문제: 왜 Multi-Armed Bandit이 필요한가?](#1-문제-왜-multi-armed-bandit이-필요한가)
2. [해결책: Thompson Sampling이란?](#2-해결책-thompson-sampling이란)
3. [실전 예제: A/B 테스팅](#3-실전-예제-ab-테스팅)
4. [언제 어떤 알고리즘을 사용할까?](#4-언제-어떤-알고리즘을-사용할까)
5. [일반적인 실수와 해결법](#5-일반적인-실수와-해결법)
6. [고급 주제](#6-고급-주제)

---

## 1. 문제: 왜 Multi-Armed Bandit이 필요한가?

### 슬롯머신 비유

카지노에 슬롯머신이 3대 있다고 상상해보세요. 각 머신은 서로 다른 확률로 당첨됩니다:

```
머신 A: 5% 당첨 확률
머신 B: 8% 당첨 확률  ← 가장 좋음 (모르는 상태)
머신 C: 3% 당첨 확률
```

**당신의 목표**: 100번 플레이해서 최대한 많이 당첨되기

**딜레마**:
- 각 머신을 충분히 테스트해야 어느 게 좋은지 알 수 있음 (탐색, Exploration)
- 하지만 테스트에만 시간을 쓰면 최고의 머신으로 돈을 못 딤 (활용, Exploitation)

이것이 바로 **Exploration-Exploitation Tradeoff**입니다!

### 실제 사례: 웹사이트 A/B 테스팅

```
시나리오: 이커머스 웹사이트 버튼 색상 테스트
- 빨간 버튼: 구매 전환율 ???
- 파란 버튼: 구매 전환율 ???
- 초록 버튼: 구매 전환율 ???

하루 방문자: 10,000명
```

**전통적인 A/B 테스팅 문제점**:

```python
# 나쁜 방법: 균등 분배
빨간 버튼: 3,333명 (전환율 2%)   → 67명 구매
파란 버튼: 3,333명 (전환율 5%)   → 167명 구매  ← 가장 좋음
초록 버튼: 3,334명 (전환율 1%)   → 33명 구매

총 구매: 267명
잃은 기회: 파란 버튼에만 집중했으면 500명 구매 가능!
```

**Multi-Armed Bandit (Thompson Sampling) 접근법**:

```python
# 좋은 방법: 적응적 분배
초반 (각각 테스트)
빨간 버튼: 500명 (전환율 2% 발견)
파란 버튼: 500명 (전환율 5% 발견) ← 좋아 보임!
초록 버튼: 500명 (전환율 1% 발견)

중반 (파란색에 더 많이 할당)
빨간 버튼: 500명
파란 버튼: 4,000명 ← 더 많이!
초록 버튼: 500명

후반 (확신을 갖고 파란색 집중)
빨간 버튼: 100명
파란 버튼: 8,700명 ← 대부분!
초록 버튼: 100명

총 구매: ~450명 (기존 대비 +68% 증가!)
```

**핵심**: Thompson Sampling은 자동으로 "학습하면서 동시에 수익 최적화"를 합니다.

---

## 2. 해결책: Thompson Sampling이란?

### 직관적 이해

Thompson Sampling은 마치 **확신의 정도**를 관리하는 것입니다:

```
처음 (불확실함):
  빨간 버튼: "전환율이 0%~10% 사이일 것 같은데... 잘 모르겠어"
  파란 버튼: "전환율이 0%~10% 사이일 것 같은데... 잘 모르겠어"

데이터 수집 후 (확신 증가):
  빨간 버튼: "전환율이 1.5%~2.5% 정도야" (좁은 범위 = 확신)
  파란 버튼: "전환율이 4.5%~5.5% 정도야" (좁은 범위 = 확신)
```

**작동 원리**:
1. 각 선택지에 대해 "예상 성능 분포"를 유지
2. 매번 결정할 때, 각 분포에서 랜덤 샘플 추출
3. 가장 높은 샘플값을 가진 선택지 선택
4. 결과를 보고 분포 업데이트

### 시각적 이해

```
확률 분포 (Probability Distribution)

초기 상태 (불확실):
빨간색  ___/‾‾‾\___      (넓음 = 불확실)
          0  5  10%

파란색  ___/‾‾‾\___      (넓음 = 불확실)
          0  5  10%

100명 테스트 후:
빨간색    _/‾\_         (좁음 = 확실)
          1 2 3%

파란색         _/‾\_    (좁음 = 확실)
              4 5 6%

Thompson Sampling 샘플링:
빨간색에서 샘플: 2.1%
파란색에서 샘플: 5.3%  ← 더 높음! 파란색 선택!
```

---

## 3. 실전 예제: A/B 테스팅

### 예제 1: 이메일 제목 테스팅 (Binary - LogisticBandit)

```python
from logisticbandit import LogisticBandit

# 1단계: 문제 정의
"""
3가지 이메일 제목 중 어떤 것이 오픈율이 가장 높을까?
- 제목A: "50% 할인!"
- 제목B: "무료 배송"
- 제목C: "오늘만 특가"
"""

# 2단계: Bandit 초기화
bandit = LogisticBandit()

# 3단계: 실험 시작 - 하루동안 데이터 수집
# (실제로는 실시간으로 업데이트하지만, 여기서는 일괄 처리)

# 아침 (각각 100명씩 테스트)
day1_morning = {
    "제목A": [100, 8],    # 100명에게 보냄, 8명 오픈
    "제목B": [100, 12],   # 100명에게 보냄, 12명 오픈
    "제목C": [100, 5],    # 100명에게 보냄, 5명 오픈
}
bandit.update(day1_morning)

# 어떤 제목이 가장 좋을까?
probs = bandit.win_prop()
print("아침 결과:")
print(f"제목A 승률: {probs['제목A']:.1%}")
print(f"제목B 승률: {probs['제목B']:.1%}")  # 가장 높을 것
print(f"제목C 승률: {probs['제목C']:.1%}")
# 출력:
# 제목A 승률: 15.2%
# 제목B 승률: 79.8%  ← 제목B가 좋아 보임!
# 제목C 승률: 5.0%

# 4단계: 결과를 바탕으로 더 많은 트래픽을 좋은 옵션에 할당
# (실전에서는 win_prop() 결과를 사용해서 자동으로 할당)

# 오후 (제목B에 더 많이 할당)
day1_afternoon = {
    "제목A": [200, 15],   # 적게 할당
    "제목B": [1000, 115], # 많이 할당!
    "제목C": [100, 4],    # 적게 할당
}
bandit.update(day1_afternoon)

# 5단계: 최종 결과
final_probs = bandit.win_prop()
print("\n최종 결과:")
for subject, prob in sorted(final_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"{subject}: {prob:.1%}")
# 출력:
# 제목B: 99.8%  ← 명확한 승자!
# 제목A: 0.2%
# 제목C: 0.0%

# 6단계: 실전 적용
best_subject = max(final_probs, key=final_probs.get)
print(f"\n✅ 결정: '{best_subject}'를 기본 제목으로 사용!")
```

**왜 좋은가?**
- ❌ 전통적 A/B 테스트: 3개를 균등하게 테스트 (1,400명 중 467명씩)
  - 예상 오픈: ~140명
- ✅ Thompson Sampling: 빠르게 최적을 찾아 집중
  - 실제 오픈: ~155명 (**+10% 개선!**)

### 예제 2: 광고 수익 최적화 (Continuous - LinearBandit)

```python
from linearbandit import LinearBandit
import numpy as np

# 1단계: 문제 정의
"""
3가지 광고 위치 중 어디가 클릭당 수익(CPC)이 가장 높을까?
- 상단 배너: 평균 CPC = ???
- 사이드바: 평균 CPC = ???
- 하단 배너: 평균 CPC = ???

CPC는 연속값 (예: $0.50, $1.23, $2.15)
"""

# 2단계: Bandit 초기화
bandit = LinearBandit(
    obs_noise=0.5,      # CPC 변동성 (표준편차)
    default_sigma=1.0   # 초기 불확실성
)

# 3단계: 첫 주 데이터 (각 위치에서 클릭 발생)
week1_data = {
    "상단_배너": [1.2, 1.5, 1.3, 1.4, 1.6],  # 5번 클릭, 평균 ~$1.40
    "사이드바": [0.8, 0.9, 0.7, 1.0, 0.85],  # 5번 클릭, 평균 ~$0.85
    "하단_배너": [2.1, 1.9, 2.3, 2.0, 2.2],  # 5번 클릭, 평균 ~$2.10
}
bandit.update(week1_data)

# 현재 상태 확인
stats = bandit.get_statistics()
print("1주차 결과:")
for position, stat in stats.items():
    print(f"{position:12s}: 평균=${stat['mu']:.2f}, "
          f"불확실성=±${stat['sigma']:.2f}, "
          f"관측={stat['count']}회")
# 출력:
# 상단_배너    : 평균=$1.35, 불확실성=±$0.19, 관측=5회
# 사이드바    : 평균=$0.83, 불확실성=±$0.19, 관측=5회
# 하단_배너    : 평균=$2.06, 불확실성=±$0.19, 관측=5회

# 승률 확인
probs = bandit.win_prop()
print("\n각 위치가 최고일 확률:")
for position, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"{position}: {prob:.1%}")
# 출력:
# 하단_배너: 99.5%  ← 거의 확실!
# 상단_배너: 0.5%
# 사이드바: 0.0%

# 4단계: 더 많은 데이터로 확신 높이기
# (하단 배너에 더 많은 트래픽 할당)
week2_data = {
    "상단_배너": [1.3, 1.4],              # 적게
    "사이드바": [0.9],                    # 매우 적게
    "하단_배너": [2.0, 2.1, 1.9, 2.2, 2.0, 2.1, 2.3]  # 많이!
}
bandit.update(week2_data)

# 5단계: 최종 통계
final_stats = bandit.get_statistics()
print("\n2주차 최종 결과:")
for position, stat in final_stats.items():
    print(f"{position:12s}: 평균=${stat['mu']:.2f}, "
          f"불확실성=±${stat['sigma']:.2f}, "
          f"총 수익=${stat['mean_reward'] * stat['count']:.2f}")

# 6단계: 실전 적용
best_position = max(probs, key=probs.get)
print(f"\n✅ 결정: 광고는 '{best_position}'에 집중 배치!")
print(f"   예상 CPC: ${final_stats[best_position]['mu']:.2f}")
```

**왜 LinearBandit을 사용했나?**
- CPC는 연속값 ($0.50, $1.23 등)
- LogisticBandit은 binary만 다룸 (클릭 O/X)
- LinearBandit은 실수값 최적화에 적합

---

## 4. 언제 어떤 알고리즘을 사용할까?

### 의사결정 플로우차트

```
질문 1: 결과가 무엇인가?
├─ Binary (성공/실패, 클릭/안클릭)
│  ├─ 시간에 따라 선호도가 변하는가?
│  │  ├─ Yes → LogisticBandit (ORTS 모드) ✅ 추천!
│  │  └─ No  → TSPar (Beta-Bernoulli)
│  └─
└─ Continuous (수익, 시간, 평점)
   └─ LinearBandit ✅

질문 2: 데이터 크기?
├─ 작음 (< 1000 obs) → decay=0.0 (과거 데이터 유지)
└─ 큼 (> 10000 obs) → decay=0.1~0.3 (최신 데이터 중시)

질문 3: 얼마나 공격적으로?
├─ 탐색 중시 → aggressive=0.5
├─ 균형 → aggressive=1.0 (기본)
└─ 활용 중시 → aggressive=2.0~5.0
```

### 실제 사용 사례 매트릭스

| 사례 | 결과 타입 | 추천 알고리즘 | 이유 |
|------|-----------|---------------|------|
| **이메일 오픈율** | Binary | LogisticBandit (ORTS) | 시간대/요일별로 선호도 변함 |
| **웹사이트 클릭** | Binary | LogisticBandit (ORTS) | 트렌드, 계절성 존재 |
| **광고 클릭** | Binary | TSPar | 빠른 의사결정, 단순함 |
| **광고 수익(CPC)** | Continuous | LinearBandit | 수익은 연속값 |
| **서버 응답시간** | Continuous | LinearBandit | 시간은 연속값, 낮을수록 좋음* |
| **상품 평점** | Continuous | LinearBandit | 평점은 연속값 (1.0~5.0) |
| **추천 시스템** | Binary | LogisticBandit | 사용자 선호도는 시간에 따라 변함 |
| **A/B 테스트 (전환)** | Binary | LogisticBandit (ORTS) | 시간 효과 고려 필요 |
| **동적 가격 책정** | Continuous | LinearBandit | 수익 최대화 |

*응답시간처럼 낮을수록 좋은 경우: 음수로 변환 (`reward = -response_time`)

---

## 5. 일반적인 실수와 해결법

### 실수 1: 너무 일찍 결론 내리기

```python
# ❌ 나쁜 예: 10명만 테스트하고 결정
bandit = LogisticBandit()
bandit.update({
    "옵션A": [10, 3],  # 30%
    "옵션B": [10, 2],  # 20%
})
# 10명은 너무 적음! 우연일 수 있음

# ✅ 좋은 예: 통계적으로 유의미한 샘플 수
bandit.update({
    "옵션A": [1000, 300],  # 30%
    "옵션B": [1000, 200],  # 20%
})
# 1000명이면 확신할 수 있음
```

**경험 법칙**:
- 최소 100명 per arm (작은 차이 감지용)
- 추천: 500-1000명 per arm (신뢰도 높음)
- Binary 이벤트에서: 최소 전환 20개 이상

### 실수 2: Decay를 잘못 사용하기

```python
# ❌ 나쁜 예: Stationary 환경에서 decay 사용
# (사용자 선호도가 변하지 않는데 decay 사용)
bandit.update(obs, decay=0.5)  # 과거 데이터를 너무 빨리 버림!

# ✅ 좋은 예 1: Stationary 환경
bandit.update(obs, decay=0.0)  # 모든 데이터 활용

# ✅ 좋은 예 2: Non-stationary 환경 (트렌드 있음)
bandit.update(obs, decay=0.1)  # 최근 데이터에 약간 더 가중치
```

**Decay 가이드라인**:
```
decay = 0.0   : 완전히 stationary (선호도 고정)
decay = 0.1   : 약간 변동 (계절성)
decay = 0.3   : 많이 변동 (트렌드)
decay = 0.5+  : 매우 많이 변동 (거의 사용 안 함)
```

### 실수 3: 잘못된 관측값 형식

```python
# LogisticBandit 사용 시
# ❌ 나쁜 예
bandit.update({
    "옵션A": [100],      # 틀림! [total, success] 필요
    "옵션B": 5,          # 틀림!
})

# ✅ 좋은 예
bandit.update({
    "옵션A": [100, 5],   # [노출 100번, 클릭 5번]
    "옵션B": [100, 8],   # [노출 100번, 클릭 8번]
})

# LinearBandit 사용 시
# ❌ 나쁜 예
bandit.update({
    "옵션A": [[1.2, 1.3, 1.4]],  # 리스트의 리스트 불필요
})

# ✅ 좋은 예
bandit.update({
    "옵션A": [1.2, 1.3, 1.4],   # 단순 리스트
    "옵션B": 2.5,                # 또는 단일 값
})
```

### 실수 4: Win Probability를 확률로 오해

```python
probs = bandit.win_prop()
# {'옵션A': 0.95, '옵션B': 0.05}

# ❌ 잘못된 해석
print("옵션A의 클릭률은 95%다")  # 틀림!

# ✅ 올바른 해석
print("옵션A가 최고일 확률이 95%다")  # 맞음!
print("95%의 경우 옵션A를 선택해야 한다")  # 맞음!
```

**Win Probability의 의미**:
- "이 옵션이 실제로 가장 좋을 확률"
- 트래픽 할당 비율로 사용 가능
- 예: 95% → 트래픽의 95%를 이 옵션에 할당

### 실수 5: 너무 많은 옵션 동시 테스트

```python
# ❌ 나쁜 예: 20개 옵션 동시 테스트
bandit.update({
    f"옵션{i}": [50, random.randint(0, 10)]
    for i in range(20)
})
# 각 옵션당 샘플이 너무 적음!

# ✅ 좋은 예: 3-5개 옵션 집중 테스트
bandit.update({
    "옵션A": [500, 25],
    "옵션B": [500, 30],
    "옵션C": [500, 20],
})
```

**권장 옵션 수**:
- 초보: 2-3개
- 적정: 3-5개
- 최대: 10개 이하
- 10개 이상이면 샘플 크기를 10배 늘려야 함

---

## 6. 고급 주제

### 6.1 실시간 트래픽 할당 시스템

```python
from logisticbandit import LogisticBandit
import time

class RealTimeBanditSystem:
    """실시간으로 트래픽을 할당하는 시스템"""

    def __init__(self):
        self.bandit = LogisticBandit()
        self.stats = {}  # 각 옵션의 노출/클릭 카운터

    def get_variant(self, user_id):
        """사용자에게 보여줄 변형 선택"""

        # 데이터가 없으면 랜덤
        if not self.bandit.get_models():
            return "variant_A"  # 기본값

        # Thompson Sampling으로 확률 계산
        probs = self.bandit.win_prop()

        # 확률에 따라 선택 (더 좋은 옵션에 더 많은 트래픽)
        import random
        r = random.random()
        cumulative = 0
        for variant, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return variant

        return list(probs.keys())[0]  # fallback

    def record_result(self, variant, clicked):
        """결과 기록 및 모델 업데이트"""

        # 카운터 초기화
        if variant not in self.stats:
            self.stats[variant] = {"views": 0, "clicks": 0}

        # 카운트 증가
        self.stats[variant]["views"] += 1
        if clicked:
            self.stats[variant]["clicks"] += 1

        # 100명마다 모델 업데이트
        total_views = sum(s["views"] for s in self.stats.values())
        if total_views % 100 == 0:
            self._update_model()

    def _update_model(self):
        """축적된 데이터로 모델 업데이트"""
        obs = {
            variant: [stats["views"], stats["clicks"]]
            for variant, stats in self.stats.items()
        }
        self.bandit.update(obs)

        # 카운터 리셋 (선택적)
        # self.stats = {}

# 사용 예제
system = RealTimeBanditSystem()

# 시뮬레이션: 10,000명의 사용자
for user_id in range(10000):
    # 1. 사용자에게 보여줄 변형 선택
    variant = system.get_variant(user_id)

    # 2. 사용자에게 보여줌 (실제로는 웹페이지 렌더링)
    # show_variant_to_user(user_id, variant)

    # 3. 사용자 행동 시뮬레이션
    true_ctrs = {"variant_A": 0.02, "variant_B": 0.05, "variant_C": 0.01}
    clicked = (random.random() < true_ctrs.get(variant, 0.02))

    # 4. 결과 기록
    system.record_result(variant, clicked)

# 최종 결과
print("최종 통계:")
for variant, stats in system.stats.items():
    ctr = stats["clicks"] / stats["views"] if stats["views"] > 0 else 0
    print(f"{variant}: {stats['views']} views, {stats['clicks']} clicks, CTR={ctr:.2%}")
```

### 6.2 비용-편익 분석

```python
from linearbandit import LinearBandit

# 시나리오: 배송 옵션 최적화
# - 빠른 배송: 고객 만족도 높지만 비용 많이 듦
# - 느린 배송: 비용 적지만 만족도 낮음

class ProfitOptimizedBandit:
    """수익을 고려한 Bandit"""

    def __init__(self):
        self.satisfaction_bandit = LinearBandit()  # 만족도 추정
        self.costs = {
            "빠른배송": 5.0,   # $5 배송비
            "표준배송": 2.0,   # $2 배송비
            "느린배송": 0.5,   # $0.5 배송비
        }

    def update(self, observations):
        """고객 만족도 데이터 업데이트"""
        # observations = {옵션: [만족도 점수 리스트]}
        self.satisfaction_bandit.update(observations)

    def get_best_option_for_profit(self, customer_value=20.0):
        """
        수익을 최대화하는 옵션 선택

        수익 = (고객 만족도 × 고객 가치) - 배송 비용
        """
        stats = self.satisfaction_bandit.get_statistics()

        profits = {}
        for option, stat in stats.items():
            satisfaction = stat['mu']  # 예상 만족도 (0-10 점)
            cost = self.costs[option]

            # 만족도를 재구매 확률로 변환 (예: 8점 = 80% 재구매)
            repurchase_prob = satisfaction / 10.0

            # 예상 수익 = 재구매 확률 × 고객 가치 - 배송 비용
            expected_profit = (repurchase_prob * customer_value) - cost

            profits[option] = expected_profit

        # 가장 수익이 높은 옵션
        best = max(profits, key=profits.get)
        return best, profits

# 사용 예제
optimizer = ProfitOptimizedBandit()

# 데이터 수집
optimizer.update({
    "빠른배송": [9.2, 9.5, 9.8, 9.1],  # 높은 만족도
    "표준배송": [7.5, 7.8, 7.2, 7.9],  # 중간 만족도
    "느린배송": [5.1, 4.8, 5.5, 5.2],  # 낮은 만족도
})

# 고객 가치별 최적 옵션
for customer_value in [10, 20, 50]:
    best, profits = optimizer.get_best_option_for_profit(customer_value)
    print(f"\n고객 가치 ${customer_value}:")
    for option, profit in sorted(profits.items(), key=lambda x: x[1], reverse=True):
        print(f"  {option:10s}: 예상 수익 ${profit:.2f}")
    print(f"  → 최선: {best}")

# 출력:
# 고객 가치 $10:
#   표준배송  : 예상 수익 $5.70
#   느린배송  : 예상 수익 $4.67
#   빠른배송  : 예상 수익 $4.25
#   → 최선: 표준배송
#
# 고객 가치 $20:
#   빠른배송  : 예상 수익 $13.72
#   표준배송  : 예상 수익 $13.10
#   느린배송  : 예상 수익 $9.84
#   → 최선: 빠른배송
```

### 6.3 Context 고려 (Contextual Bandit의 시작)

```python
# 간단한 방법: Context별로 별도 Bandit 유지
class ContextualBanditSimple:
    """문맥별로 다른 Bandit을 유지하는 간단한 방법"""

    def __init__(self):
        self.bandits = {}  # context -> Bandit 매핑

    def get_variant(self, user_context):
        """사용자 문맥에 맞는 변형 선택"""

        # 문맥 식별 (예: 모바일 vs 데스크탑)
        context_key = user_context["device_type"]

        # 해당 문맥의 Bandit 가져오기 (없으면 생성)
        if context_key not in self.bandits:
            from logisticbandit import LogisticBandit
            self.bandits[context_key] = LogisticBandit()

        bandit = self.bandits[context_key]

        # 해당 Bandit에서 선택
        if bandit.get_models():
            probs = bandit.win_prop()
            # ... (확률에 따라 선택)
        else:
            return "default_variant"

    def update(self, context_key, observations):
        """해당 문맥의 Bandit 업데이트"""
        if context_key in self.bandits:
            self.bandits[context_key].update(observations)

# 사용 예제
contextual = ContextualBanditSimple()

# 모바일 사용자
mobile_variant = contextual.get_variant({"device_type": "mobile"})
# ... 결과 수집 후
contextual.update("mobile", {"variant_A": [100, 15]})

# 데스크탑 사용자 (별도 학습!)
desktop_variant = contextual.get_variant({"device_type": "desktop"})
# ... 결과 수집 후
contextual.update("desktop", {"variant_A": [100, 8]})
```

---

## 다음 단계

1. **예제 실행해보기**:
   ```bash
   python examples/linear_bandit.py
   python examples/ab_testing.py
   ```

2. **본인의 데이터로 실험**:
   - 작은 데이터셋으로 시작
   - 시뮬레이션으로 검증
   - 실제 트래픽에 적용

3. **더 읽을거리**:
   - [PERFORMANCE.md](PERFORMANCE.md): 성능 최적화
   - [API Reference](README.md#api-reference): 상세 API 문서
   - [Examples](examples/): 더 많은 예제

4. **도움이 필요하면**:
   - GitHub Issues: 버그 리포트 및 질문
   - 문서: 이 튜토리얼과 README.md

---

**핵심 요약**:
- Multi-Armed Bandit = "학습하면서 동시에 최적화"
- Thompson Sampling = 확률적으로 최선의 선택 찾기
- LogisticBandit = Binary 결과 (클릭/전환)
- LinearBandit = Continuous 결과 (수익/시간)
- 충분한 데이터 수집 후 결정!
