"""
통합 워크플로우: statlingua + vitals 결합
실제 업무 시나리오
"""

from anthropic import Anthropic
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
import os

# 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from statlingua_py.explainer import StatLinguaExplainer
from vitals_py.task import Task
from vitals_py.scorers import model_graded_qa


def workflow_1_weekly_mentoring_automation():
    """
    워크플로우 1: 주간 멘토링 자료 자동 생성 및 품질 관리
    """
    print("="*70)
    print("워크플로우 1: 주간 멘토링 자료 생성 및 품질 평가")
    print("="*70)

    client = Anthropic()

    # 12주 커리큘럼
    curriculum = pd.DataFrame({
        'week': range(1, 13),
        'topic': [
            "Python 기초 문법",
            "자료구조 (리스트, 딕셔너리)",
            "제어문 (if, for, while)",
            "함수와 모듈",
            "파일 입출력",
            "예외 처리",
            "객체지향 프로그래밍",
            "Pandas 기초",
            "데이터 시각화",
            "웹 스크래핑",
            "API 활용",
            "프로젝트 통합"
        ],
        'level': [
            'beginner', 'beginner', 'beginner', 'intermediate',
            'intermediate', 'intermediate', 'intermediate', 'intermediate',
            'advanced', 'advanced', 'advanced', 'advanced'
        ]
    })

    # 특정 주차 자료 생성 (예: 5주차)
    week = 5
    topic = curriculum.loc[curriculum['week'] == week, 'topic'].values[0]
    level = curriculum.loc[curriculum['week'] == week, 'level'].values[0]

    print(f"\n📚 {week}주차: {topic} (난이도: {level})")

    # 1. 자료 생성
    prompt = f"""
주제: {topic}
대상: 대학생 초보자
난이도: {level}

다음 형식으로 학습 자료를 만들어주세요:

1. 핵심 개념 (3-5문장)
2. 실생활 비유
3. 코드 예제 (주석 포함)
4. 연습 문제 3개 (힌트 포함)
"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    material = response.content[0].text

    # 자료 저장
    output_dir = Path(os.path.dirname(__file__)) / "mentoring_materials"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"week{week}_{topic.replace(' ', '_')}.md", 'w', encoding='utf-8') as f:
        f.write(f"# {week}주차: {topic}\n\n")
        f.write(material)

    print(f"✓ 자료 생성 완료: {output_dir / f'week{week}_{topic.replace(\" \", \"_\")}.md'}")

    # 2. 품질 평가 (vitals 사용)
    quality_dataset = pd.DataFrame({
        'input': [f"{topic}을 초보자에게 설명하세요"],
        'target': ["명확한 개념, 비유, 예제, 연습문제 포함"]
    })

    def material_solver(inputs, **kwargs):
        # 이미 생성된 자료 사용
        return {
            'result': [material],
            'solver_metadata': [{'generated': True}]
        }

    quality_task = Task(
        dataset=quality_dataset,
        solver=material_solver,
        scorer=model_graded_qa(
            client=client,
            instructions="""
교육 자료의 품질을 평가하세요:
1. 개념이 명확한가?
2. 비유가 적절한가?
3. 예제가 실행 가능한가?
4. 연습문제가 적절한가?
"""
        ),
        name=f"week{week}_quality"
    )

    quality_task.eval()

    print(f"\n품질 평가 점수: {quality_task.metrics.accuracy:.0%}")

    return material


def workflow_2_research_report_automation():
    """
    워크플로우 2: 연구 보고서 자동 생성 (OLED)
    """
    print("\n" + "="*70)
    print("워크플로우 2: OLED 연구 보고서 자동화")
    print("="*70)

    # OLED 실험 데이터 시뮬레이션
    np.random.seed(2026)
    n = 60

    experiments = {
        'tandem_efficiency': pd.DataFrame({
            'efficiency': 25 + 5*np.random.randn(n),
            'layer1_nm': 50 + 10*np.random.randn(n),
            'layer2_nm': 40 + 8*np.random.randn(n),
            'dopant_pct': 2 + 0.3*np.random.randn(n)
        }),
        'reliability': pd.DataFrame({
            'lifetime_hours': 10000 + 2000*np.random.randn(n),
            'temperature_C': 80 + 10*np.random.randn(n),
            'humidity_pct': 50 + 10*np.random.randn(n)
        }),
        'birefringence': pd.DataFrame({
            'crack_prob': np.random.beta(2, 5, n),
            'laser_power_W': 100 + 20*np.random.randn(n),
            'glass_thickness_um': 500 + 50*np.random.randn(n)
        })
    }

    client = Anthropic()
    explainer = StatLinguaExplainer(client=client)

    report_sections = []

    for exp_name, data in experiments.items():
        print(f"\n분석 중: {exp_name}")

        # 통계 분석
        y_col = data.columns[0]
        X_cols = data.columns[1:]

        X = sm.add_constant(data[X_cols])
        model = sm.OLS(data[y_col], X)
        results = model.fit()

        # 경영진용 요약
        exec_summary = explainer.explain(
            results,
            context=f"""
실험: {exp_name}
목표: 2026년 양산 준비
중요성: 차세대 OLED 패널 경쟁력 확보
""",
            audience="manager",
            verbosity="brief"
        )

        # 연구진용 상세
        tech_detail = explainer.explain(
            results,
            context=f"실험: {exp_name}",
            audience="researcher",
            verbosity="detailed"
        )

        report_sections.append({
            'experiment': exp_name,
            'executive': exec_summary.text,
            'technical': tech_detail.text
        })

    # 보고서 생성
    report_path = Path(os.path.dirname(__file__)) / "research_reports"
    report_path.mkdir(exist_ok=True)

    # 경영진용
    with open(report_path / "executive_summary.md", 'w', encoding='utf-8') as f:
        f.write("# OLED 연구 진행 보고 (경영진용)\n\n")
        f.write(f"작성일: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        for section in report_sections:
            f.write(f"## {section['experiment']}\n\n")
            f.write(section['executive'])
            f.write("\n\n---\n\n")

    # 연구진용
    with open(report_path / "technical_report.md", 'w', encoding='utf-8') as f:
        f.write("# OLED 연구 상세 보고 (연구진용)\n\n")
        f.write(f"작성일: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        for section in report_sections:
            f.write(f"## {section['experiment']}\n\n")
            f.write(section['technical'])
            f.write("\n\n---\n\n")

    print(f"\n✓ 보고서 생성 완료:")
    print(f"  - {report_path / 'executive_summary.md'}")
    print(f"  - {report_path / 'technical_report.md'}")


def workflow_3_explanation_quality_evaluation():
    """
    워크플로우 3: 통계 설명의 품질 평가
    statlingua로 생성한 설명을 vitals로 평가
    """
    print("\n" + "="*70)
    print("워크플로우 3: 통계 설명 품질 평가")
    print("="*70)

    # 여러 통계 모델
    np.random.seed(100)

    models_data = {
        'simple_regression': {
            'data': pd.DataFrame({
                'y': np.random.randn(50),
                'x': np.random.randn(50)
            }),
            'description': "단순 선형회귀"
        },
        'multiple_regression': {
            'data': pd.DataFrame({
                'y': np.random.randn(50),
                'x1': np.random.randn(50),
                'x2': np.random.randn(50),
                'x3': np.random.randn(50)
            }),
            'description': "다중 선형회귀"
        }
    }

    client = Anthropic()
    explainer = StatLinguaExplainer(client=client)

    # 설명 생성
    explanations = {}

    for name, info in models_data.items():
        data = info['data']
        y_col = 'y'
        X_cols = [c for c in data.columns if c != 'y']

        X = sm.add_constant(data[X_cols])
        model = sm.OLS(data[y_col], X)
        results = model.fit()

        # novice용 설명 생성
        explanation = explainer.explain(
            results,
            audience="novice",
            verbosity="moderate"
        )

        explanations[name] = explanation.text

    # vitals로 품질 평가
    eval_dataset = pd.DataFrame({
        'input': [
            f"{info['description']} 결과를 초보자에게 설명하세요"
            for info in models_data.values()
        ],
        'target': [
            "명확성, 정확성, 접근성을 모두 갖춘 설명"
            for _ in models_data
        ]
    })

    def explanation_solver(inputs, **kwargs):
        # 이미 생성된 설명 사용
        return {
            'result': list(explanations.values()),
            'solver_metadata': [{'pre_generated': True}] * len(explanations)
        }

    quality_task = Task(
        dataset=eval_dataset,
        solver=explanation_solver,
        scorer=model_graded_qa(
            client=client,
            instructions="""
통계 설명의 품질을 평가하세요:
1. 통계적으로 정확한가?
2. 초보자가 이해하기 쉬운가?
3. 핵심 인사이트를 포함하는가?
4. 실무적 의미를 설명하는가?
"""
        ),
        name="explanation_quality"
    )

    quality_task.eval(view=True)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("경고: ANTHROPIC_API_KEY 환경변수를 설정하세요")
        print("사용법: export ANTHROPIC_API_KEY='your-api-key'")
        exit(1)

    # 실행
    workflow_1_weekly_mentoring_automation()
    workflow_2_research_report_automation()
    workflow_3_explanation_quality_evaluation()

    print("\n" + "="*70)
    print("모든 워크플로우 완료!")
    print("="*70)
