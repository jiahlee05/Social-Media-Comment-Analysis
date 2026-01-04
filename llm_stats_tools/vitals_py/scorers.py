"""
Vitals Scorers: 다양한 평가 방식
"""

from typing import List, Dict, Callable, Optional, Literal
import re
from anthropic import Anthropic


# Scorer 반환 타입
ScorerResult = Dict[str, List]


def detect_includes(case_sensitive: bool = False) -> Callable:
    """
    target이 result에 포함되는지 확인

    Args:
        case_sensitive: 대소문자 구분 여부

    Returns:
        Scorer 함수
    """
    def scorer(samples: List) -> ScorerResult:
        scores = []

        for sample in samples:
            result = sample.result or ""
            target = sample.target or ""

            if not case_sensitive:
                result = result.lower()
                target = target.lower()

            if target in result:
                scores.append('C')  # Correct
            else:
                scores.append('I')  # Incorrect

        return {'score': scores}

    return scorer


def detect_match(
    location: Literal["begin", "end", "any", "exact"] = "end",
    case_sensitive: bool = False
) -> Callable:
    """
    위치 기반 매칭

    Args:
        location: 매칭 위치
        case_sensitive: 대소문자 구분

    Returns:
        Scorer 함수
    """
    def scorer(samples: List) -> ScorerResult:
        scores = []

        for sample in samples:
            result = (sample.result or "").strip()
            target = (sample.target or "").strip()

            if not case_sensitive:
                result = result.lower()
                target = target.lower()

            matched = False
            if location == "begin":
                matched = result.startswith(target)
            elif location == "end":
                matched = result.endswith(target)
            elif location == "any":
                matched = target in result
            elif location == "exact":
                matched = result == target

            scores.append('C' if matched else 'I')

        return {'score': scores}

    return scorer


def detect_pattern(
    pattern: str,
    case_sensitive: bool = False,
    all_must_match: bool = False
) -> Callable:
    """
    정규표현식 패턴 매칭

    Args:
        pattern: 정규표현식 패턴
        case_sensitive: 대소문자 구분
        all_must_match: 모든 매치가 target에 있어야 하는지

    Returns:
        Scorer 함수
    """
    def scorer(samples: List) -> ScorerResult:
        scores = []
        flags = 0 if case_sensitive else re.IGNORECASE

        for sample in samples:
            result = sample.result or ""
            target = sample.target or ""

            matches = re.findall(pattern, result, flags=flags)

            if not matches:
                scores.append('I')
            elif all_must_match:
                all_in_target = all(m in target for m in matches)
                scores.append('C' if all_in_target else 'I')
            else:
                any_in_target = any(m in target for m in matches)
                scores.append('C' if any_in_target else 'I')

        return {'score': scores}

    return scorer


def detect_exact(case_sensitive: bool = False) -> Callable:
    """정확한 매칭"""
    return detect_match(location="exact", case_sensitive=case_sensitive)


def model_graded_qa(
    client: Optional[Anthropic] = None,
    model: str = "claude-sonnet-4-5-20250929",
    instructions: Optional[str] = None,
    partial_credit: bool = False
) -> Callable:
    """
    LLM을 사용한 QA 평가

    Args:
        client: Anthropic client
        model: 사용할 모델
        instructions: 평가 지침
        partial_credit: 부분 점수 허용 여부

    Returns:
        Scorer 함수
    """
    if client is None:
        client = Anthropic()

    default_instructions = """
당신은 공정한 평가자입니다.
주어진 질문(INPUT), 모범 답안(TARGET), 그리고 학생의 답변(RESULT)을 보고
답변의 정확성을 평가하세요.

평가 기준:
- GRADE: C (Correct, 정답)
- GRADE: P (Partially Correct, 부분 정답)
- GRADE: I (Incorrect, 오답)

반드시 "GRADE: " 형식으로 시작하는 한 줄로 평가를 제시하세요.
"""

    if instructions:
        grading_prompt = default_instructions + "\n추가 지침:\n" + instructions
    else:
        grading_prompt = default_instructions

    def scorer(samples: List) -> ScorerResult:
        scores = []
        metadata = []

        for sample in samples:
            prompt = f"""
{grading_prompt}

INPUT: {sample.input}
TARGET: {sample.target}
RESULT: {sample.result}

평가해주세요:
"""

            response = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # GRADE 추출
            grade_match = re.search(r'GRADE\s*:\s*([CPI])', response_text, re.IGNORECASE)

            if grade_match:
                grade = grade_match.group(1).upper()
                if not partial_credit and grade == 'P':
                    grade = 'I'  # 부분 점수 불허
                scores.append(grade)
            else:
                scores.append('I')  # 파싱 실패 시 오답 처리

            metadata.append({
                'grading_response': response_text,
                'model': model
            })

        return {
            'score': scores,
            'scorer_metadata': metadata
        }

    return scorer


def model_graded_fact(
    client: Optional[Anthropic] = None,
    model: str = "claude-sonnet-4-5-20250929",
    instructions: Optional[str] = None
) -> Callable:
    """
    LLM을 사용한 사실 확인 평가

    Args:
        client: Anthropic client
        model: 사용할 모델
        instructions: 평가 지침

    Returns:
        Scorer 함수
    """
    default_instructions = """
당신은 사실 확인 평가자입니다.
주어진 핵심 사실(TARGET)이 답변(RESULT)에 정확히 포함되어 있는지 평가하세요.

평가 기준:
- GRADE: C - 사실이 정확히 포함됨
- GRADE: I - 사실이 누락되었거나 부정확함

반드시 "GRADE: " 형식으로 시작하는 한 줄로 평가를 제시하세요.
"""

    if instructions:
        grading_prompt = default_instructions + "\n추가 지침:\n" + instructions
    else:
        grading_prompt = default_instructions

    def scorer(samples: List) -> ScorerResult:
        scores = []
        metadata = []

        for sample in samples:
            prompt = f"""
{grading_prompt}

INPUT: {sample.input}
TARGET (확인할 사실): {sample.target}
RESULT (답변): {sample.result}

평가해주세요:
"""

            response = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # GRADE 추출
            grade_match = re.search(r'GRADE\s*:\s*([CI])', response_text, re.IGNORECASE)

            if grade_match:
                scores.append(grade_match.group(1).upper())
            else:
                scores.append('I')

            metadata.append({
                'grading_response': response_text,
                'model': model
            })

        return {
            'score': scores,
            'scorer_metadata': metadata
        }

    return scorer
