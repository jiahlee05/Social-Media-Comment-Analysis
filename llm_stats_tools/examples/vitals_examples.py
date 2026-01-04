"""
Vitals Python 사용 예제
"""

from anthropic import Anthropic
import pandas as pd
import os
import sys

# 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vitals_py.task import Task, generate
from vitals_py.scorers import (
    detect_includes, detect_match, detect_pattern,
    model_graded_qa, model_graded_fact
)


def example_1_basic_evaluation():
    """기본 평가 예제"""
    print("="*60)
    print("예제 1: 기본 LLM 평가")
    print("="*60)

    # 평가 데이터셋
    dataset = pd.DataFrame({
        'input': [
            "2 + 2는?",
            "Python에서 리스트의 첫 번째 요소는?",
            "지구에서 가장 높은 산은?"
        ],
        'target': [
            "4",
            "list[0]",
            "에베레스트"
        ]
    })

    # Client
    client = Anthropic()

    # Task 생성
    task = Task(
        dataset=dataset,
        solver=generate(client),
        scorer=detect_includes(),
        name="basic_qa"
    )

    # 평가 실행
    task.eval(view=True)

    # 결과 확인
    results = task.get_samples()
    print("\n결과:")
    print(results)


def example_2_model_graded():
    """모델 기반 평가"""
    print("\n" + "="*60)
    print("예제 2: 모델 기반 평가")
    print("="*60)

    dataset = pd.DataFrame({
        'input': [
            "Python의 주요 특징을 설명하세요",
            "머신러닝과 딥러닝의 차이는?",
            "REST API란 무엇인가요?"
        ],
        'target': [
            "간결한 문법, 동적 타이핑, 풍부한 라이브러리, 인터프리터 언어",
            "머신러닝은 넓은 개념, 딥러닝은 신경망 기반의 머신러닝 하위 분야",
            "HTTP 기반 아키텍처 스타일, 상태 없음, 자원 중심"
        ]
    })

    client = Anthropic()

    task = Task(
        dataset=dataset,
        solver=generate(client),
        scorer=model_graded_qa(
            client=client,
            partial_credit=True,
            instructions="기술적 정확성과 완성도를 평가하세요"
        ),
        name="technical_qa"
    )

    task.eval(view=True)


def example_3_python_coding_eval():
    """Python 코딩 능력 평가"""
    print("\n" + "="*60)
    print("예제 3: Python 코딩 능력 평가")
    print("="*60)

    dataset = pd.DataFrame({
        'input': [
            "리스트 컴프리헨션으로 1-10의 제곱 리스트를 만드세요",
            "딕셔너리 두 개를 병합하는 코드를 작성하세요",
            "문자열을 역순으로 뒤집는 함수를 작성하세요",
            "리스트에서 중복을 제거하는 방법은?"
        ],
        'target': [
            "[x**2 for x in range(1, 11)] 또는 유사한 컴프리헨션",
            "{**dict1, **dict2} 또는 dict1.update(dict2)",
            "string[::-1] 또는 ''.join(reversed(string))",
            "list(set(my_list)) 또는 dict.fromkeys()"
        ]
    })

    client = Anthropic()

    # 코딩 전문 평가 프롬프트
    def coding_solver_with_system(inputs: list, **kwargs):
        results = []
        metadata = []

        for inp in inputs:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                system="당신은 Python 전문가입니다. 간결하고 효율적인 코드를 작성하세요.",
                messages=[{"role": "user", "content": inp}]
            )

            results.append(response.content[0].text)
            metadata.append({'usage': response.usage.model_dump()})

        return {
            'result': results,
            'solver_metadata': metadata
        }

    task = Task(
        dataset=dataset,
        solver=coding_solver_with_system,
        scorer=model_graded_qa(
            client=client,
            instructions="""
Python 코드를 평가할 때 다음을 고려하세요:
1. 문법적 정확성
2. Pythonic한 스타일
3. 효율성
4. 가독성
"""
        ),
        name="python_coding",
        epochs=2  # 일관성 확인을 위해 2번 반복
    )

    task.eval(view=True)

    print(f"\n일관성 분석 (2 epochs):")
    results = task.get_samples()
    print(results[['input', 'epoch', 'score']])


def example_4_model_comparison():
    """여러 모델 비교"""
    print("\n" + "="*60)
    print("예제 4: 모델 비교 평가")
    print("="*60)

    dataset = pd.DataFrame({
        'input': [
            "R에서 데이터프레임의 첫 5행을 보는 함수는?",
            "결측치를 제거하는 R 함수는?",
            "두 변수의 상관계수를 구하는 R 함수는?"
        ],
        'target': [
            "head()",
            "na.omit() 또는 complete.cases()",
            "cor()"
        ]
    })

    client = Anthropic()

    models = [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001"
    ]

    tasks = {}

    for model_name in models:
        print(f"\n--- 평가 중: {model_name} ---")

        def solver_factory(model):
            def solver(inputs, **kwargs):
                results = []
                metadata = []

                for inp in inputs:
                    response = client.messages.create(
                        model=model,
                        max_tokens=500,
                        messages=[{"role": "user", "content": inp}]
                    )

                    results.append(response.content[0].text)
                    metadata.append({
                        'model': model,
                        'usage': response.usage.model_dump()
                    })

                return {
                    'result': results,
                    'solver_metadata': metadata
                }
            return solver

        task = Task(
            dataset=dataset,
            solver=solver_factory(model_name),
            scorer=detect_includes(),
            name=f"r_qa_{model_name.split('-')[1]}"  # sonnet, haiku
        )

        task.eval(view=False)
        tasks[model_name] = task

    # 결과 비교
    print("\n" + "="*60)
    print("모델 성능 비교")
    print("="*60)

    for model_name, task in tasks.items():
        print(f"\n{model_name}:")
        print(f"  정확도: {task.metrics.accuracy:.2%}")
        print(f"  정답: {task.metrics.correct}/{task.metrics.total_samples}")


def example_5_student_mentoring():
    """학생 멘토링 자료 품질 평가"""
    print("\n" + "="*60)
    print("예제 5: 학생 멘토링 자료 품질 평가")
    print("="*60)

    # 학생들이 자주 묻는 질문
    dataset = pd.DataFrame({
        'input': [
            "Python 리스트와 튜플의 차이점을 초보자에게 설명해주세요",
            "for 루프와 while 루프는 언제 사용하나요?",
            "함수의 return 문은 왜 필요한가요?",
            "변수 이름을 지을 때 주의할 점은?"
        ],
        'target': [
            "리스트는 변경 가능(mutable), 튜플은 변경 불가(immutable). 예제와 비유 포함",
            "반복 횟수를 알 때 for, 조건에 따라 반복할 때 while. 실용적 예제 포함",
            "함수에서 결과를 반환하여 재사용. print와의 차이 설명",
            "의미 있는 이름, snake_case, 예약어 피하기"
        ]
    })

    client = Anthropic()

    # 교육용 프롬프트
    def educational_solver(inputs, **kwargs):
        results = []
        metadata = []

        for inp in inputs:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                system="""
당신은 대학생들을 가르치는 친절한 멘토입니다.
- 초보자 수준에 맞춰 설명하세요
- 실생활 비유를 사용하세요
- 간단한 코드 예제를 포함하세요
- 격려하는 톤을 유지하세요
""",
                messages=[{"role": "user", "content": inp}]
            )

            results.append(response.content[0].text)
            metadata.append({'usage': response.usage.model_dump()})

        return {
            'result': results,
            'solver_metadata': metadata
        }

    task = Task(
        dataset=dataset,
        solver=educational_solver,
        scorer=model_graded_qa(
            client=client,
            partial_credit=True,
            instructions="""
교육 자료로서의 품질을 평가하세요:
1. 초보자가 이해하기 쉬운가?
2. 실용적인 예제가 있는가?
3. 격려하는 톤인가?
4. 핵심 개념을 정확히 설명하는가?
"""
        ),
        name="student_mentoring_quality"
    )

    task.eval(view=True)

    # 상세 결과 출력
    results = task.get_samples()
    for idx, row in results.iterrows():
        print(f"\n질문: {row['input'][:50]}...")
        print(f"점수: {row['score']}")
        print(f"답변 샘플: {row['result'][:200]}...")


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("경고: ANTHROPIC_API_KEY 환경변수를 설정하세요")
        print("사용법: export ANTHROPIC_API_KEY='your-api-key'")
        exit(1)

    # 로그 디렉토리 설정
    os.environ['VITALS_LOG_DIR'] = './vitals_logs'

    example_1_basic_evaluation()
    example_2_model_graded()
    example_3_python_coding_eval()
    example_4_model_comparison()
    example_5_student_mentoring()

    print("\n" + "="*60)
    print("모든 예제 완료!")
    print(f"로그 위치: {os.environ['VITALS_LOG_DIR']}")
    print("="*60)
