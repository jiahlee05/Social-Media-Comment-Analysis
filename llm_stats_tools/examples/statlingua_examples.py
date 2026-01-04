"""
StatLingua Python 사용 예제
"""

from anthropic import Anthropic
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
import sys
import os

# 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from statlingua_py.explainer import StatLinguaExplainer, explain


def example_1_basic_regression():
    """기본 선형회귀 예제"""
    print("="*60)
    print("예제 1: 기본 선형회귀 분석")
    print("="*60)

    # 데이터 생성
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(n)

    # statsmodels로 회귀분석
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    # 설명 생성
    client = Anthropic()  # API_KEY는 환경변수에서

    explanation = explain(
        results,
        client=client,
        context="""
        이 분석은 두 개의 예측변수가 결과변수에 미치는 영향을 조사합니다.
        데이터는 시뮬레이션된 것으로, 교육 목적으로 사용됩니다.
        """,
        audience="student",
        verbosity="moderate"
    )

    print(explanation.text)
    print("\n" + "="*60 + "\n")


def example_2_multiple_audiences():
    """다양한 청중을 위한 설명"""
    print("="*60)
    print("예제 2: 청중별 맞춤 설명")
    print("="*60)

    # mtcars 스타일 데이터
    data = pd.DataFrame({
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4],
        'wt': [2.62, 2.88, 2.32, 3.21, 3.44, 3.46, 3.57, 3.19],
        'hp': [110, 110, 93, 110, 175, 105, 245, 62]
    })

    # 회귀분석
    X = sm.add_constant(data[['wt', 'hp']])
    model = sm.OLS(data['mpg'], X)
    results = model.fit()

    client = Anthropic()
    explainer = StatLinguaExplainer(client=client)

    # 여러 청중을 위한 설명
    audiences = ["novice", "manager", "researcher"]

    for audience in audiences:
        print(f"\n--- {audience.upper()}용 설명 ---\n")

        explanation = explainer.explain(
            results,
            context="자동차 무게와 마력이 연비에 미치는 영향 분석",
            audience=audience,
            verbosity="moderate"
        )

        print(explanation.text)
        print("\n" + "-"*60 + "\n")


def example_3_oled_research():
    """OLED 연구 시뮬레이션"""
    print("="*60)
    print("예제 3: OLED Tandem 구조 최적화")
    print("="*60)

    # OLED 실험 데이터 시뮬레이션
    np.random.seed(2025)
    n = 50

    data = pd.DataFrame({
        'efficiency': 25 + 5*np.random.randn(n),
        'layer1_thickness': 50 + 20*np.random.randn(n),
        'layer2_thickness': 40 + 15*np.random.randn(n),
        'dopant_concentration': 2 + 0.5*np.random.randn(n),
        'annealing_temp': 150 + 10*np.random.randn(n)
    })

    # 효율에 영향 주는 요인 분석
    X = sm.add_constant(data[[
        'layer1_thickness',
        'layer2_thickness',
        'dopant_concentration',
        'annealing_temp'
    ]])
    model = sm.OLS(data['efficiency'], X)
    results = model.fit()

    client = Anthropic()

    # 연구진용 상세 분석
    print("\n[연구진용 상세 분석]\n")
    tech_explanation = explain(
        results,
        client=client,
        context="""
        Tandem OLED 구조 최적화 연구
        목표: 발광 효율 30% 향상
        제약조건: 총 두께 10μm 이하 유지
        2026년 SID Display Week 발표 예정
        """,
        audience="researcher",
        verbosity="detailed"
    )
    print(tech_explanation.text)

    # 경영진용 요약
    print("\n\n[경영진용 요약]\n")
    exec_explanation = explain(
        results,
        client=client,
        context="""
        차세대 OLED 패널 성능 최적화
        양산 목표: 2026년
        기대효과: 에너지 효율 30% 개선
        """,
        audience="manager",
        verbosity="brief"
    )
    print(exec_explanation.text)


def example_4_batch_processing():
    """배치 처리 - 여러 모델 동시 설명"""
    print("="*60)
    print("예제 4: 배치 처리 (10개 모델)")
    print("="*60)

    client = Anthropic()
    explainer = StatLinguaExplainer(client=client)

    # 10개 다른 데이터셋/모델
    models = {}

    for i in range(10):
        np.random.seed(i)
        X = np.random.randn(50, 2)
        y = i*X[:, 0] + (10-i)*X[:, 1] + np.random.randn(50)

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const)
        models[f'experiment_{i+1}'] = model.fit()

    # 모든 모델 설명 생성
    explanations = {}

    for name, result in models.items():
        print(f"\n처리 중: {name}")

        explanation = explainer.explain(
            result,
            context=f"실험 {name}의 결과 분석",
            audience="researcher",
            verbosity="brief"
        )

        explanations[name] = explanation
        print(f"✓ 완료")

    # 결과 저장
    output_file = os.path.join(os.path.dirname(__file__), 'batch_explanations.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, exp in explanations.items():
            f.write(f"## {name}\n\n")
            f.write(exp.text)
            f.write("\n\n---\n\n")

    print(f"\n총 {len(explanations)}개 모델 설명 완료")
    print(f"결과 저장: {output_file}")


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("경고: ANTHROPIC_API_KEY 환경변수를 설정하세요")
        print("사용법: export ANTHROPIC_API_KEY='your-api-key'")
        exit(1)

    example_1_basic_regression()
    example_2_multiple_audiences()
    example_3_oled_research()
    example_4_batch_processing()
