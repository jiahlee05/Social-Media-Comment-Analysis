# LLM Stats Tools

R 패키지 `statlingua`와 `vitals`의 핵심 기능을 Python으로 완전 재구현한 라이브러리입니다.

## 개요

이 프로젝트는 두 가지 강력한 도구를 제공합니다:

1. **statlingua_py**: LLM을 사용하여 통계 분석 결과를 자연어로 설명
2. **vitals_py**: LLM 평가 프레임워크 - 모델 성능을 체계적으로 측정

## 프로젝트 구조

```
llm_stats_tools/
├── statlingua_py/
│   ├── __init__.py
│   └── explainer.py
├── vitals_py/
│   ├── __init__.py
│   ├── task.py
│   └── scorers.py
├── examples/
│   ├── statlingua_examples.py
│   ├── vitals_examples.py
│   └── integrated_workflow.py
└── requirements.txt
```

## 설치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd llm_stats_tools
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. API 키 설정

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## StatLingua: 통계 설명 자동화

### 기능

- 통계 모델 결과를 자연어로 자동 설명
- 다양한 청중 레벨 지원 (novice, student, researcher, manager, domain_expert)
- 여러 출력 형식 (markdown, html, json, text, latex)
- statsmodels, scikit-learn, scipy 지원

### 빠른 시작

```python
from anthropic import Anthropic
import statsmodels.api as sm
from statlingua_py.explainer import explain

# 모델 fitting
X = sm.add_constant(X_data)
model = sm.OLS(y_data, X)
results = model.fit()

# 설명 생성
client = Anthropic()
explanation = explain(
    results,
    client=client,
    context="자동차 무게와 마력이 연비에 미치는 영향",
    audience="novice",
    verbosity="moderate"
)

print(explanation.text)
```

### 청중별 설명

```python
from statlingua_py.explainer import StatLinguaExplainer

explainer = StatLinguaExplainer(client=client)

# 초보자용
novice_exp = explainer.explain(results, audience="novice")

# 경영진용
manager_exp = explainer.explain(results, audience="manager", verbosity="brief")

# 연구자용
researcher_exp = explainer.explain(results, audience="researcher", verbosity="detailed")
```

## Vitals: LLM 평가 프레임워크

### 기능

- 체계적인 LLM 평가 파이프라인
- 다양한 scorer 지원
  - `detect_includes`: 포함 여부 확인
  - `detect_match`: 위치 기반 매칭
  - `detect_pattern`: 정규표현식 매칭
  - `model_graded_qa`: LLM 기반 QA 평가
  - `model_graded_fact`: LLM 기반 사실 확인
- Epoch 반복을 통한 일관성 검증
- JSON 로깅 및 메트릭 추적

### 빠른 시작

```python
from anthropic import Anthropic
import pandas as pd
from vitals_py.task import Task, generate
from vitals_py.scorers import detect_includes

# 평가 데이터셋
dataset = pd.DataFrame({
    'input': [
        "2 + 2는?",
        "Python에서 리스트의 첫 번째 요소는?",
    ],
    'target': [
        "4",
        "list[0]",
    ]
})

# Task 생성 및 평가
client = Anthropic()
task = Task(
    dataset=dataset,
    solver=generate(client),
    scorer=detect_includes(),
    name="basic_qa"
)

task.eval(view=True)
```

### 모델 기반 평가

```python
from vitals_py.scorers import model_graded_qa

task = Task(
    dataset=technical_questions,
    solver=generate(client),
    scorer=model_graded_qa(
        client=client,
        partial_credit=True,
        instructions="기술적 정확성과 완성도를 평가하세요"
    )
)

task.eval()
print(f"정확도: {task.metrics.accuracy:.2%}")
```

### 모델 비교

```python
models = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]

for model_name in models:
    task = Task(
        dataset=dataset,
        solver=generate(client, model=model_name),
        scorer=detect_includes(),
        name=f"eval_{model_name}"
    )
    task.eval()
    print(f"{model_name}: {task.metrics.accuracy:.2%}")
```

## 통합 워크플로우

StatLingua와 Vitals를 결합하여 강력한 워크플로우를 구성할 수 있습니다.

### 예제 1: 멘토링 자료 자동 생성 및 품질 평가

```python
from statlingua_py.explainer import StatLinguaExplainer
from vitals_py.task import Task
from vitals_py.scorers import model_graded_qa

# 1. 자료 생성
client = Anthropic()
material = generate_teaching_material(topic="Python 기초")

# 2. 품질 평가
quality_task = Task(
    dataset=quality_dataset,
    solver=lambda inputs: {'result': [material]},
    scorer=model_graded_qa(
        client=client,
        instructions="교육 자료의 품질을 평가하세요"
    )
)
quality_task.eval()
```

### 예제 2: 연구 보고서 자동화

```python
# 통계 분석
results = fit_statistical_model(data)

# 청중별 설명 생성
explainer = StatLinguaExplainer(client=client)

executive_summary = explainer.explain(
    results,
    audience="manager",
    verbosity="brief"
)

technical_report = explainer.explain(
    results,
    audience="researcher",
    verbosity="detailed"
)
```

## 예제 실행

### StatLingua 예제

```bash
cd examples
python statlingua_examples.py
```

예제 포함:
- 기본 선형회귀 분석
- 청중별 맞춤 설명
- OLED 연구 시뮬레이션
- 배치 처리 (10개 모델)

### Vitals 예제

```bash
cd examples
python vitals_examples.py
```

예제 포함:
- 기본 LLM 평가
- 모델 기반 평가
- Python 코딩 능력 평가
- 모델 비교
- 학생 멘토링 자료 품질 평가

### 통합 워크플로우

```bash
cd examples
python integrated_workflow.py
```

워크플로우 포함:
- 주간 멘토링 자료 자동 생성 및 품질 관리
- OLED 연구 보고서 자동화
- 통계 설명 품질 평가

## API 레퍼런스

### StatLingua

#### `explain(model_result, ...)`

통계 모델 결과를 설명합니다.

**매개변수:**
- `model_result`: 통계 모델 결과 객체 (statsmodels, sklearn, scipy)
- `client`: Anthropic client 객체
- `api_key`: API 키 (client가 없을 때)
- `context`: 추가 컨텍스트 정보
- `audience`: 대상 청중 ("novice", "student", "researcher", "manager", "domain_expert")
- `verbosity`: 상세도 ("brief", "moderate", "detailed")
- `style`: 출력 형식 ("markdown", "html", "json", "text", "latex")
- `model`: 사용할 Claude 모델

**반환값:**
- `StatLinguaExplanation` 객체

### Vitals

#### `Task(dataset, solver, scorer, ...)`

LLM 평가 Task를 생성합니다.

**매개변수:**
- `dataset`: 'input'과 'target' 컬럼을 가진 DataFrame
- `solver`: 입력을 받아 결과를 생성하는 함수
- `scorer`: 결과를 평가하는 함수
- `name`: Task 이름 (선택)
- `epochs`: 반복 횟수 (기본값: 1)
- `log_dir`: 로그 저장 디렉토리 (선택)

**메서드:**
- `solve(**kwargs)`: Solver 실행
- `score(**kwargs)`: Scorer 실행
- `measure()`: 메트릭 계산
- `log()`: 결과 로깅
- `eval(view=False, **kwargs)`: 전체 파이프라인 실행

#### Scorers

- `detect_includes(case_sensitive=False)`: 포함 여부 확인
- `detect_match(location="end", case_sensitive=False)`: 위치 기반 매칭
- `detect_pattern(pattern, case_sensitive=False)`: 정규표현식 매칭
- `detect_exact(case_sensitive=False)`: 정확한 매칭
- `model_graded_qa(client, model, instructions, partial_credit=False)`: LLM 기반 QA 평가
- `model_graded_fact(client, model, instructions)`: LLM 기반 사실 확인

## 환경 변수

- `ANTHROPIC_API_KEY`: Anthropic API 키 (필수)
- `VITALS_LOG_DIR`: Vitals 로그 디렉토리 (기본값: `./vitals_logs`)

## 라이선스

MIT License

## 기여

기여는 언제나 환영합니다! Issue나 Pull Request를 자유롭게 제출해주세요.

## 문의

문제가 발생하거나 질문이 있으시면 Issue를 열어주세요.

## 참고

이 프로젝트는 다음 R 패키지들에서 영감을 받았습니다:
- [statlingua](https://github.com/anthropics/statlingua)
- [vitals](https://github.com/anthropics/vitals)
