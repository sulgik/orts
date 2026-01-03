"""
Step-by-Step Tutorial: Multi-Armed Bandit 입문

이 파일은 Multi-Armed Bandit을 처음 배우는 사람들을 위한
단계별 튜토리얼입니다. 각 단계를 실행하면서 개념을 이해하세요.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from logisticbandit import LogisticBandit
from linearbandit import LinearBandit


def print_section(title):
    """섹션 구분자 출력"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def tutorial_part1_basic_concept():
    """Part 1: 기본 개념 - 왜 Multi-Armed Bandit인가?"""
    print_section("Part 1: 왜 Multi-Armed Bandit이 필요한가?")

    print("""
상황: 웹사이트에 3가지 버튼 색상 중 어떤 게 더 많은 클릭을 받을까요?
- 빨간 버튼 (진짜 CTR: 2%)
- 파란 버튼 (진짜 CTR: 5%)  ← 가장 좋음! (하지만 모르는 상태)
- 초록 버튼 (진짜 CTR: 1%)

총 1,000명의 방문자가 있습니다.

옵션 1: 균등 분배 (전통적 A/B 테스트)
  빨간색: 333명 → 7명 클릭
  파란색: 333명 → 17명 클릭
  초록색: 334명 → 3명 클릭
  총 클릭: 27명

옵션 2: Multi-Armed Bandit (Thompson Sampling)
  처음엔 각각 테스트 → 빠르게 파란색이 좋다는 걸 발견
  → 파란색에 더 많은 트래픽 할당
  총 클릭: 45명 (67% 향상!)
    """)

    input("\n계속하려면 Enter를 누르세요...")


def tutorial_part2_first_bandit():
    """Part 2: 첫 번째 Bandit 만들기"""
    print_section("Part 2: 첫 번째 Bandit 만들어보기")

    print("\n단계 1: LogisticBandit 생성하기")
    print("-" * 70)
    print("코드:")
    print("  from logisticbandit import LogisticBandit")
    print("  bandit = LogisticBandit()")
    print()

    bandit = LogisticBandit()
    print("✓ Bandit이 생성되었습니다!")

    print("\n단계 2: 초기 데이터 입력하기")
    print("-" * 70)
    print("3가지 버튼에 각각 100명씩 노출시켰습니다:")
    print("  빨간색: 100명 중 2명 클릭")
    print("  파란색: 100명 중 5명 클릭")
    print("  초록색: 100명 중 1명 클릭")
    print()
    print("코드:")
    print('  bandit.update({')
    print('      "빨간색": [100, 2],   # [노출수, 클릭수]')
    print('      "파란색": [100, 5],')
    print('      "초록색": [100, 1],')
    print('  })')
    print()

    bandit.update({
        "빨간색": [100, 2],
        "파란색": [100, 5],
        "초록색": [100, 1],
    })
    print("✓ 데이터가 입력되었습니다!")

    print("\n단계 3: 어떤 버튼이 가장 좋을까요?")
    print("-" * 70)
    print("코드:")
    print("  probs = bandit.win_prop()")
    print("  for color, prob in probs.items():")
    print('      print(f"{color}: {prob:.1%}")')
    print()

    probs = bandit.win_prop()
    print("결과:")
    for color in ["빨간색", "파란색", "초록색"]:
        if color in probs:
            print(f"  {color}: {probs[color]:6.1%} ← 이 버튼이 최고일 확률")

    print(f"\n해석: 파란색이 최고일 확률이 {probs['파란색']:.0%}입니다!")
    print("      하지만 아직 데이터가 적어서 100% 확신하기는 이릅니다.")

    input("\n계속하려면 Enter를 누르세요...")

    return bandit


def tutorial_part3_more_data(bandit):
    """Part 3: 더 많은 데이터 수집하기"""
    print_section("Part 3: 더 많은 데이터로 확신 높이기")

    print("\n이제 파란색이 좋아 보이니, 더 많은 트래픽을 할당해봅시다.")
    print()
    print("새로운 데이터:")
    print("  빨간색: 100명 추가 → 2명 클릭")
    print("  파란색: 500명 추가 → 25명 클릭 (더 많이 테스트!)")
    print("  초록색: 100명 추가 → 1명 클릭")
    print()

    bandit.update({
        "빨간색": [100, 2],
        "파란색": [500, 25],
        "초록색": [100, 1],
    })

    probs = bandit.win_prop()
    print("업데이트 후 결과:")
    for color in ["빨간색", "파란색", "초록색"]:
        if color in probs:
            print(f"  {color}: {probs[color]:6.1%}")

    print(f"\n이제 파란색이 최고일 확률이 {probs['파란색']:.0%}로 증가했습니다!")
    print("확신이 높아졌으니, 파란색을 주력으로 사용해도 좋습니다.")

    input("\n계속하려면 Enter를 누르세요...")


def tutorial_part4_continuous_rewards():
    """Part 4: 연속값 보상 - LinearBandit"""
    print_section("Part 4: 연속값 최적화하기 (LinearBandit)")

    print("""
이번에는 다른 문제를 풀어봅시다.

상황: 3가지 광고 위치 중 어디가 클릭당 수익(CPC)이 가장 높을까요?
- 상단 배너: CPC = ???
- 사이드바: CPC = ???
- 하단 배너: CPC = ???

이전과의 차이:
- 이전: 클릭 여부 (O/X) → Binary → LogisticBandit
- 지금: 수익 금액 ($1.20, $2.50 등) → Continuous → LinearBandit
    """)

    print("\n단계 1: LinearBandit 생성")
    print("-" * 70)
    print("코드:")
    print("  bandit = LinearBandit(obs_noise=0.3)")
    print()

    bandit = LinearBandit(obs_noise=0.3)
    print("✓ LinearBandit이 생성되었습니다!")

    print("\n단계 2: 클릭 데이터 입력 (각 클릭의 수익)")
    print("-" * 70)
    print("각 위치에서 5번씩 클릭이 발생했고, 각 클릭의 수익은:")
    print("  상단 배너: [$1.20, $1.35, $1.15, $1.40, $1.30]")
    print("  사이드바: [$0.80, $0.95, $0.75, $0.85, $0.90]")
    print("  하단 배너: [$2.10, $2.25, $1.95, $2.30, $2.15]")
    print()

    obs = {
        "상단_배너": [1.20, 1.35, 1.15, 1.40, 1.30],
        "사이드바": [0.80, 0.95, 0.75, 0.85, 0.90],
        "하단_배너": [2.10, 2.25, 1.95, 2.30, 2.15],
    }

    bandit.update(obs)
    print("✓ 데이터가 입력되었습니다!")

    print("\n단계 3: 각 위치의 예상 수익 확인")
    print("-" * 70)
    print("코드:")
    print("  stats = bandit.get_statistics()")
    print()

    stats = bandit.get_statistics()
    print("결과:")
    print(f"{'위치':12s}  {'예상 CPC':>10s}  {'불확실성':>10s}  {'클릭 수':>8s}")
    print("-" * 70)
    for position in ["상단_배너", "사이드바", "하단_배너"]:
        if position in stats:
            s = stats[position]
            print(f"{position:12s}  ${s['mu']:>9.2f}  ±${s['sigma']:>8.2f}  {s['count']:>8d}회")

    print("\n단계 4: 어느 위치가 가장 좋을까요?")
    print("-" * 70)
    probs = bandit.win_prop()
    print("각 위치가 최고일 확률:")
    for position in ["상단_배너", "사이드바", "하단_배너"]:
        if position in probs:
            print(f"  {position:12s}: {probs[position]:6.1%}")

    best_position = max(probs, key=probs.get)
    print(f"\n✅ 결론: '{best_position}'에 광고를 배치하세요!")
    print(f"   예상 CPC: ${stats[best_position]['mu']:.2f}")

    input("\n계속하려면 Enter를 누르세요...")


def tutorial_part5_exploration_exploitation():
    """Part 5: Exploration vs Exploitation 이해하기"""
    print_section("Part 5: 탐색과 활용의 균형")

    print("""
Multi-Armed Bandit의 핵심 딜레마:

1. 탐색 (Exploration)
   - 새로운 옵션을 시도해서 정보를 얻기
   - 단기적으로는 손해일 수 있음
   - 장기적으로 더 좋은 옵션을 찾을 수 있음

2. 활용 (Exploitation)
   - 현재 가장 좋아 보이는 옵션 사용
   - 단기적으로는 이득
   - 더 좋은 옵션을 놓칠 수 있음

Thompson Sampling은 이 균형을 자동으로 맞춰줍니다!
    """)

    print("\n시나리오: 2가지 옵션을 1,000명에게 테스트")
    print("-" * 70)

    # 진짜 CTR (사용자는 모름)
    true_ctrs = {"옵션A": 0.03, "옵션B": 0.05}

    bandit = LogisticBandit()
    stats_over_time = []

    # 시뮬레이션
    np.random.seed(42)
    batch_size = 50
    for batch in range(20):  # 20 배치 = 1,000명
        # Thompson Sampling으로 트래픽 할당
        if bandit.get_models():
            probs = bandit.win_prop()
        else:
            probs = {"옵션A": 0.5, "옵션B": 0.5}

        # 트래픽 할당
        n_a = int(batch_size * probs.get("옵션A", 0.5))
        n_b = batch_size - n_a

        # 실제 클릭 시뮬레이션
        clicks_a = np.random.binomial(n_a, true_ctrs["옵션A"])
        clicks_b = np.random.binomial(n_b, true_ctrs["옵션B"])

        # 업데이트
        bandit.update({
            "옵션A": [n_a, clicks_a],
            "옵션B": [n_b, clicks_b],
        })

        # 기록
        stats_over_time.append({
            "batch": batch + 1,
            "n_a": n_a,
            "n_b": n_b,
            "prob_a": probs.get("옵션A", 0.5),
            "prob_b": probs.get("옵션B", 0.5),
        })

    # 결과 출력
    print("\n시간에 따른 트래픽 할당 변화:")
    print(f"{'배치':>5s}  {'옵션A 할당':>12s}  {'옵션B 할당':>12s}  {'옵션B 우월성':>15s}")
    print("-" * 70)

    for i in [0, 4, 9, 14, 19]:  # 일부만 출력
        s = stats_over_time[i]
        batch_num = (i + 1) * batch_size
        print(f"{batch_num:5d}명  {s['n_a']:5d}명 ({s['prob_a']:4.1%})  "
              f"{s['n_b']:5d}명 ({s['prob_b']:4.1%})  "
              f"{s['prob_b']:6.1%}")

    print("""
패턴 관찰:
- 처음엔 50:50으로 균등 분배 (탐색)
- 시간이 지나면서 옵션B가 더 좋다는 걸 발견
- 점점 더 많은 트래픽을 옵션B에 할당 (활용)
- 하지만 옵션A도 완전히 무시하지는 않음 (계속 탐색)

이것이 Thompson Sampling의 아름다움입니다!
    """)

    input("\n계속하려면 Enter를 누르세요...")


def tutorial_part6_common_parameters():
    """Part 6: 중요한 매개변수 이해하기"""
    print_section("Part 6: 중요한 매개변수들")

    print("""
1. draw (Monte Carlo 샘플 수)
   - 기본값: 100,000
   - 높을수록: 정확하지만 느림
   - 낮을수록: 빠르지만 부정확

2. aggressive (공격성)
   - 기본값: 1.0 (균형)
   - < 1.0: 탐색 중시 (더 보수적)
   - > 1.0: 활용 중시 (더 공격적)

3. decay (감쇠)
   - 기본값: 0.0 (과거 데이터 유지)
   - 0.1-0.3: 최근 데이터에 더 가중치
   - > 0.5: 과거 데이터를 빠르게 망각
    """)

    print("\n실험: aggressive 매개변수의 효과")
    print("-" * 70)

    bandit = LogisticBandit()
    bandit.update({
        "옵션A": [100, 5],   # 5% CTR
        "옵션B": [100, 8],   # 8% CTR (더 좋음)
    })

    print("\n같은 데이터, 다른 aggressive 값:")
    print(f"{'aggressive':>12s}  {'옵션A 확률':>12s}  {'옵션B 확률':>12s}")
    print("-" * 70)

    for agg in [0.5, 1.0, 2.0, 5.0]:
        probs = bandit.win_prop(aggressive=agg)
        print(f"{agg:12.1f}  {probs['옵션A']:12.1%}  {probs['옵션B']:12.1%}")

    print("""
관찰:
- aggressive가 높을수록 옵션B에 더 많은 트래픽 집중
- aggressive가 낮을수록 옵션A도 계속 테스트
- 기본값 1.0은 좋은 균형점

언제 조정할까?
- aggressive 낮추기 (0.5): 초기 탐색 단계, 확신이 낮을 때
- aggressive 높이기 (2.0): 충분한 데이터 확보 후, 빠른 수익화 필요
    """)

    input("\n계속하려면 Enter를 누르세요...")


def tutorial_part7_real_world_tips():
    """Part 7: 실전 팁"""
    print_section("Part 7: 실전 사용 팁")

    print("""
✅ 해야 할 것들:

1. 충분한 데이터 수집
   - 최소: 각 옵션당 100회 이상
   - 권장: 각 옵션당 500-1000회
   - Binary 이벤트: 최소 전환 20개 이상

2. 통계적 유의성 확인
   - win_prop()이 95% 이상일 때 확신 가능
   - 80-95%는 좀 더 데이터 수집 고려

3. 주기적으로 모니터링
   - 매일 또는 매주 결과 확인
   - 이상 패턴 감지

4. 환경에 맞게 설정
   - Stationary: decay=0.0
   - Non-stationary: decay=0.1-0.3

❌ 하지 말아야 할 것들:

1. 너무 일찍 결론 내리기
   - 10-20명만으로는 부족!
   - 최소 100명은 테스트하세요

2. 너무 많은 옵션 동시 테스트
   - 3-5개가 적당
   - 10개 이상은 샘플 크기를 크게 늘려야 함

3. 잘못된 데이터 형식
   - LogisticBandit: [total, success]
   - LinearBandit: [reward1, reward2, ...] 또는 single value

4. Win probability를 잘못 해석
   - 95% = "이 옵션이 최고일 확률"
   - 95% ≠ "이 옵션의 성능이 95%"

📊 실전 체크리스트:

□ 옵션 수가 3-5개인가?
□ 각 옵션당 최소 100회 이상 데이터가 있는가?
□ Binary vs Continuous를 올바르게 선택했는가?
□ Win probability가 80% 이상인가?
□ 결과를 주기적으로 모니터링하는가?
    """)

    input("\n튜토리얼을 종료하려면 Enter를 누르세요...")


def main():
    """튜토리얼 메인 함수"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    Multi-Armed Bandit & Thompson Sampling                        ║
║    Step-by-Step Tutorial                                         ║
║                                                                  ║
║    이 튜토리얼은 초보자를 위한 단계별 가이드입니다.              ║
║    각 단계를 천천히 따라가며 개념을 이해하세요.                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    input("시작하려면 Enter를 누르세요...")

    # Part 1: 기본 개념
    tutorial_part1_basic_concept()

    # Part 2: 첫 Bandit
    bandit = tutorial_part2_first_bandit()

    # Part 3: 더 많은 데이터
    tutorial_part3_more_data(bandit)

    # Part 4: 연속값
    tutorial_part4_continuous_rewards()

    # Part 5: Exploration vs Exploitation
    tutorial_part5_exploration_exploitation()

    # Part 6: 매개변수
    tutorial_part6_common_parameters()

    # Part 7: 실전 팁
    tutorial_part7_real_world_tips()

    print_section("축하합니다! 튜토리얼을 완료했습니다!")

    print("""
🎉 이제 여러분은 Multi-Armed Bandit의 기초를 마스터했습니다!

다음 단계:
1. examples/ 디렉토리의 다른 예제들 실행해보기
2. TUTORIAL.md 읽어보기 (더 자세한 설명)
3. 본인의 데이터로 실험해보기

질문이 있으면:
- README.md의 API Reference 참고
- GitHub Issues에 질문 올리기

Happy experimenting! 🚀
    """)


if __name__ == "__main__":
    main()
