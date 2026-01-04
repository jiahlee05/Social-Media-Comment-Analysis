"""
Vitals for Python: LLM Evaluation Framework
"""

from typing import Optional, Callable, Dict, List, Any, Literal
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from anthropic import Anthropic
import json
import os
from datetime import datetime
from pathlib import Path
import uuid


@dataclass
class Sample:
    """평가 샘플"""
    id: str
    input: str
    target: str
    epoch: int = 1
    result: Optional[str] = None
    score: Optional[str] = None
    solver_metadata: Optional[Dict] = None
    scorer_metadata: Optional[Dict] = None


@dataclass
class TaskMetrics:
    """평가 메트릭"""
    accuracy: float = 0.0
    partial_credit_rate: float = 0.0
    total_samples: int = 0
    correct: int = 0
    partial: int = 0
    incorrect: int = 0


class Task:
    """
    LLM 평가 Task

    R vitals 패키지의 Python 구현
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        solver: Callable,
        scorer: Callable,
        name: Optional[str] = None,
        epochs: int = 1,
        log_dir: Optional[str] = None
    ):
        """
        Args:
            dataset: 'input'과 'target' 컬럼을 가진 DataFrame
            solver: 입력을 받아 결과를 생성하는 함수
            scorer: 결과를 평가하는 함수
            name: Task 이름
            epochs: 각 샘플을 반복할 횟수
            log_dir: 로그 저장 디렉토리
        """
        self.dataset = dataset
        self.solver_func = solver
        self.scorer_func = scorer
        self.name = name or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.epochs = epochs
        self.log_dir = log_dir or os.getenv("VITALS_LOG_DIR", "./vitals_logs")

        self.samples: List[Sample] = []
        self.metrics: Optional[TaskMetrics] = None
        self._solver_executed = False
        self._scorer_executed = False

        # 로그 디렉토리 생성
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def solve(self, **kwargs) -> 'Task':
        """Solver 실행"""
        print(f"🔧 Solving task: {self.name}")
        print(f"   Samples: {len(self.dataset)} × {self.epochs} epochs")

        # epoch별로 샘플 확장
        expanded_samples = []
        for epoch in range(1, self.epochs + 1):
            for idx, row in self.dataset.iterrows():
                sample = Sample(
                    id=f"{idx}_{epoch}",
                    input=row['input'],
                    target=row['target'],
                    epoch=epoch
                )
                expanded_samples.append(sample)

        # Solver 실행
        inputs = [s.input for s in expanded_samples]
        solver_results = self.solver_func(inputs, **kwargs)

        # 결과 저장
        for i, sample in enumerate(expanded_samples):
            sample.result = solver_results['result'][i]
            if 'solver_metadata' in solver_results:
                sample.solver_metadata = solver_results['solver_metadata'][i]

        self.samples = expanded_samples
        self._solver_executed = True

        print(f"   ✓ Solved {len(self.samples)} samples")
        return self

    def score(self, **kwargs) -> 'Task':
        """Scorer 실행"""
        if not self._solver_executed:
            raise ValueError("solve()를 먼저 실행하세요")

        print(f"📊 Scoring task: {self.name}")

        # Scorer 실행
        scorer_results = self.scorer_func(self.samples, **kwargs)

        # 결과 저장
        for i, sample in enumerate(self.samples):
            sample.score = scorer_results['score'][i]
            if 'scorer_metadata' in scorer_results:
                sample.scorer_metadata = scorer_results['scorer_metadata'][i]

        self._scorer_executed = True
        print(f"   ✓ Scored {len(self.samples)} samples")
        return self

    def measure(self) -> 'Task':
        """메트릭 계산"""
        if not self._scorer_executed:
            raise ValueError("score()를 먼저 실행하세요")

        scores = [s.score for s in self.samples]

        correct = sum(1 for s in scores if s == 'C')
        partial = sum(1 for s in scores if s == 'P')
        incorrect = sum(1 for s in scores if s == 'I')
        total = len(scores)

        self.metrics = TaskMetrics(
            accuracy=correct / total if total > 0 else 0.0,
            partial_credit_rate=partial / total if total > 0 else 0.0,
            total_samples=total,
            correct=correct,
            partial=partial,
            incorrect=incorrect
        )

        print(f"\n📈 Metrics:")
        print(f"   Accuracy: {self.metrics.accuracy:.2%}")
        print(f"   Correct: {correct}/{total}")
        print(f"   Partial: {partial}/{total}")
        print(f"   Incorrect: {incorrect}/{total}")

        return self

    def log(self) -> str:
        """결과를 JSON으로 로깅"""
        log_file = Path(self.log_dir) / f"{self.name}_{uuid.uuid4().hex[:8]}.json"

        log_data = {
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "epochs": self.epochs,
            "metrics": self.metrics.__dict__ if self.metrics else None,
            "samples": [
                {
                    "id": s.id,
                    "input": s.input,
                    "target": s.target,
                    "result": s.result,
                    "score": s.score,
                    "epoch": s.epoch,
                    "solver_metadata": s.solver_metadata,
                    "scorer_metadata": s.scorer_metadata
                }
                for s in self.samples
            ]
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Logged to: {log_file}")
        return str(log_file)

    def get_samples(self) -> pd.DataFrame:
        """샘플을 DataFrame으로 반환"""
        return pd.DataFrame([
            {
                'id': s.id,
                'epoch': s.epoch,
                'input': s.input,
                'target': s.target,
                'result': s.result,
                'score': s.score
            }
            for s in self.samples
        ])

    def eval(self, view: bool = False, **kwargs) -> 'Task':
        """전체 평가 파이프라인 실행"""
        self.solve(**kwargs)
        self.score(**kwargs)
        self.measure()
        self.log()

        if view:
            self.view()

        return self

    def view(self):
        """결과 시각화 (간단한 출력)"""
        print("\n" + "="*70)
        print(f"Task: {self.name}")
        print("="*70)

        df = self.get_samples()
        print(df.to_string())

        if self.metrics:
            print(f"\n{self.metrics}")


# Solver 생성 함수들
def generate(client: Anthropic, model: str = "claude-sonnet-4-5-20250929"):
    """
    기본 generate solver

    Args:
        client: Anthropic client
        model: 사용할 Claude 모델

    Returns:
        Solver 함수
    """
    def solver(inputs: List[str], **kwargs) -> Dict[str, List]:
        results = []
        metadata = []

        for inp in inputs:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
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
