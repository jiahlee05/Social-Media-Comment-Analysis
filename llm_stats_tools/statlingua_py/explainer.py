"""
StatLingua for Python: LLM-powered statistical output explanation
"""

import anthropic
from anthropic import Anthropic
from typing import Optional, Literal, Dict, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json


AudienceType = Literal["novice", "student", "researcher", "manager", "domain_expert"]
VerbosityType = Literal["brief", "moderate", "detailed"]
StyleType = Literal["markdown", "html", "json", "text", "latex"]


@dataclass
class StatLinguaExplanation:
    """설명 결과를 담는 데이터 클래스"""
    text: str
    model_type: str
    audience: AudienceType
    verbosity: VerbosityType
    style: StyleType
    raw_response: Optional[Dict[str, Any]] = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"StatLinguaExplanation(model_type='{self.model_type}', audience='{self.audience}')"


class StatLinguaExplainer:
    """통계 모델 결과를 자연어로 설명하는 클래스"""

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929"
    ):
        """
        Args:
            client: Anthropic client 객체
            api_key: Anthropic API 키 (client가 없을 때 사용)
            model: 사용할 Claude 모델
        """
        if client is None:
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = client
        self.model = model

    def explain(
        self,
        model_result,
        context: Optional[str] = None,
        audience: AudienceType = "novice",
        verbosity: VerbosityType = "moderate",
        style: StyleType = "markdown",
        **kwargs
    ) -> StatLinguaExplanation:
        """
        통계 모델 결과를 설명

        Args:
            model_result: 통계 모델 결과 (statsmodels, sklearn 등)
            context: 추가 컨텍스트 정보
            audience: 대상 청중
            verbosity: 설명 상세도
            style: 출력 형식

        Returns:
            StatLinguaExplanation 객체
        """
        # 모델 타입 감지
        model_type = self._detect_model_type(model_result)

        # 모델 결과 요약
        summary = self._summarize_model(model_result, model_type)

        # 프롬프트 생성
        prompt = self._build_prompt(
            summary=summary,
            model_type=model_type,
            context=context,
            audience=audience,
            verbosity=verbosity,
            style=style
        )

        # LLM 호출
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        explanation_text = response.content[0].text

        return StatLinguaExplanation(
            text=explanation_text,
            model_type=model_type,
            audience=audience,
            verbosity=verbosity,
            style=style,
            raw_response=response.model_dump()
        )

    def _detect_model_type(self, model_result) -> str:
        """모델 타입 감지"""
        class_name = type(model_result).__name__
        module_name = type(model_result).__module__

        # statsmodels
        if "statsmodels" in module_name:
            if "OLS" in class_name or "RegressionResults" in class_name:
                return "linear_regression"
            elif "Logit" in class_name or "LogitResults" in class_name:
                return "logistic_regression"
            elif "GLM" in class_name:
                return "glm"
            elif "MixedLM" in class_name:
                return "mixed_effects"

        # scikit-learn
        elif "sklearn" in module_name:
            if "LinearRegression" in class_name:
                return "linear_regression"
            elif "LogisticRegression" in class_name:
                return "logistic_regression"
            elif "RandomForest" in class_name:
                return "random_forest"
            elif "GradientBoosting" in class_name:
                return "gradient_boosting"

        # scipy
        elif "scipy" in module_name:
            if "TtestResult" in class_name:
                return "t_test"
            elif "Chi2ContingencyResult" in class_name:
                return "chi_square"

        return "unknown"

    def _summarize_model(self, model_result, model_type: str) -> str:
        """모델 결과를 텍스트로 요약"""

        if model_type in ["linear_regression", "logistic_regression", "glm"]:
            return self._summarize_regression(model_result)
        elif model_type in ["random_forest", "gradient_boosting"]:
            return self._summarize_sklearn_model(model_result)
        elif model_type == "t_test":
            return self._summarize_ttest(model_result)
        elif model_type == "chi_square":
            return self._summarize_chi2(model_result)
        else:
            return str(model_result)

    def _summarize_regression(self, model_result) -> str:
        """회귀분석 결과 요약"""
        try:
            # statsmodels 결과
            summary_text = f"""
Model Type: {type(model_result).__name__}

Coefficients:
{model_result.summary().tables[1].as_text()}

Model Statistics:
R-squared: {model_result.rsquared:.4f}
Adj. R-squared: {model_result.rsquared_adj:.4f}
F-statistic: {model_result.fvalue:.4f}
Prob (F-statistic): {model_result.f_pvalue:.4e}
AIC: {model_result.aic:.2f}
BIC: {model_result.bic:.2f}

Number of Observations: {model_result.nobs}
Df Residuals: {model_result.df_resid}
Df Model: {model_result.df_model}
"""
            return summary_text
        except AttributeError:
            # sklearn 결과
            return f"Model: {type(model_result).__name__}\nCoefficients available via model attributes"

    def _summarize_sklearn_model(self, model_result) -> str:
        """sklearn 모델 요약"""
        summary = f"Model Type: {type(model_result).__name__}\n\n"

        if hasattr(model_result, 'feature_importances_'):
            summary += "Feature Importances:\n"
            for i, imp in enumerate(model_result.feature_importances_):
                summary += f"  Feature {i}: {imp:.4f}\n"

        if hasattr(model_result, 'score'):
            summary += "\nModel has scoring capability\n"

        return summary

    def _summarize_ttest(self, result) -> str:
        """t-검정 결과 요약"""
        return f"""
T-test Results:
Test Statistic: {result.statistic:.4f}
P-value: {result.pvalue:.4e}
Degrees of Freedom: {result.df if hasattr(result, 'df') else 'N/A'}
"""

    def _summarize_chi2(self, result) -> str:
        """카이제곱 검정 결과 요약"""
        return f"""
Chi-Square Test Results:
Test Statistic: {result.statistic:.4f}
P-value: {result.pvalue:.4e}
Degrees of Freedom: {result.dof}
"""

    def _build_prompt(
        self,
        summary: str,
        model_type: str,
        context: Optional[str],
        audience: AudienceType,
        verbosity: VerbosityType,
        style: StyleType
    ) -> str:
        """설명 생성을 위한 프롬프트 구성"""

        audience_descriptions = {
            "novice": "통계 배경이 제한적인 초보자",
            "student": "통계를 배우는 학생",
            "researcher": "강한 통계 배경과 방법론에 익숙한 연구자",
            "manager": "의사결정을 위한 고수준 인사이트가 필요한 관리자",
            "domain_expert": "자신의 분야 전문가이지만 통계 전문가는 아닌 사람"
        }

        verbosity_descriptions = {
            "brief": "간략한 요약",
            "moderate": "균형잡힌 설명",
            "detailed": "포괄적이고 상세한 해석"
        }

        style_instructions = {
            "markdown": "일반 마크다운 형식으로 작성",
            "html": "HTML 프래그먼트 형식으로 작성",
            "json": "JSON 문자열로 구조화하여 작성",
            "text": "일반 텍스트로 작성",
            "latex": "LaTeX 프래그먼트로 작성"
        }

        prompt = f"""다음 통계 분석 결과를 설명해주세요.

모델 타입: {model_type}

분석 결과:
{summary}

{f'추가 컨텍스트: {context}' if context else ''}

대상 청중: {audience_descriptions[audience]}
상세도: {verbosity_descriptions[verbosity]}
출력 형식: {style_instructions[style]}

다음 요소를 포함하여 설명해주세요:
1. 주요 발견사항
2. 통계적 유의성 해석
3. 실무적 의미
4. 주의사항이나 한계점

{audience}에게 적합하고 {verbosity} 수준으로 작성해주세요.
"""

        return prompt


# 편의 함수
def explain(
    model_result,
    client: Optional[Anthropic] = None,
    api_key: Optional[str] = None,
    context: Optional[str] = None,
    audience: AudienceType = "novice",
    verbosity: VerbosityType = "moderate",
    style: StyleType = "markdown",
    model: str = "claude-sonnet-4-5-20250929"
) -> StatLinguaExplanation:
    """
    통계 모델 결과 설명 (단축 함수)
    """
    explainer = StatLinguaExplainer(client=client, api_key=api_key, model=model)
    return explainer.explain(
        model_result=model_result,
        context=context,
        audience=audience,
        verbosity=verbosity,
        style=style
    )
