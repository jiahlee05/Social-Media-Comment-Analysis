# Session 5: 실험 데이터 처리 & 보고서 작성

**수업 시간:** 2시간
**목표:** 전체 분석 파이프라인 구축 및 자동화된 보고서 생성

---

## 📋 수업 목차

1. **실험 데이터 클리닝** (30분)
2. **종합 분석 파이프라인** (30min)
3. **자동화된 보고서 생성** (40분)
4. **최종 프로젝트** (20분)

---

## 1. 실험 데이터 클리닝 (30분)

### 실험 데이터의 특징

**커뮤니케이션 실험 예시:**
- A/B 테스트 (광고 문구, 디자인)
- 메시지 프레이밍 효과
- 미디어 노출 영향
- 사용자 인터페이스 테스트

### 흔한 데이터 문제

1. **결측값 (Missing Data)**
2. **이상치 (Outliers)**
3. **중복 데이터**
4. **데이터 타입 불일치**
5. **불일관한 카테고리 이름**

### 결측값 처리

```python
import pandas as pd
import numpy as np

df = pd.read_csv('experiment_data.csv')

# 결측값 확인
print(df.isnull().sum())
print(f"결측값 비율: {df.isnull().sum().sum() / df.size * 100:.2f}%")

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Pattern')
plt.show()

# 처리 방법
# 1. 삭제
df_clean = df.dropna()  # 결측값 있는 행 삭제
df_clean = df.dropna(subset=['important_column'])  # 특정 컬럼만

# 2. 대체
df['age'].fillna(df['age'].mean(), inplace=True)  # 평균으로
df['category'].fillna(df['category'].mode()[0], inplace=True)  # 최빈값으로

# 3. 보간
df['score'].interpolate(method='linear', inplace=True)
```

### 이상치 탐지 및 처리

```python
# IQR 방법
Q1 = df['response_time'].quantile(0.25)
Q3 = df['response_time'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 범위
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 식별
outliers = df[(df['response_time'] < lower_bound) | (df['response_time'] > upper_bound)]
print(f"이상치 개수: {len(outliers)}")

# 처리 옵션
# 1. 제거
df_no_outliers = df[(df['response_time'] >= lower_bound) & (df['response_time'] <= upper_bound)]

# 2. 캡핑 (Capping)
df['response_time'] = df['response_time'].clip(lower_bound, upper_bound)

# 3. 변환 (Log transformation)
df['response_time_log'] = np.log1p(df['response_time'])
```

### 데이터 검증

```python
# 범위 확인
assert df['age'].min() >= 0, "나이는 음수일 수 없습니다"
assert df['age'].max() <= 120, "나이가 비현실적입니다"

# 카테고리 확인
valid_conditions = ['Control', 'Treatment_A', 'Treatment_B']
assert df['condition'].isin(valid_conditions).all(), "잘못된 조건값 발견"

# 중복 확인
duplicates = df.duplicated(subset=['participant_id'], keep=False)
if duplicates.any():
    print(f"⚠️ 중복 참가자 {duplicates.sum()}명 발견")
```

---

## 2. 종합 분석 파이프라인 (30분)

### 분석 파이프라인 구조

```
데이터 로드 → 클리닝 → 탐색 → 분석 → 시각화 → 보고서
```

### 파이프라인 클래스 구현

```python
class ExperimentAnalysisPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        self.df = pd.read_csv(self.data_path)
        print(f"✅ 데이터 로드 완료: {self.df.shape}")
        return self

    def clean_data(self):
        """데이터 클리닝"""
        # 결측값 처리
        self.df.dropna(subset=['participant_id', 'condition'], inplace=True)

        # 이상치 제거
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[(self.df[col] >= Q1 - 1.5*IQR) &
                             (self.df[col] <= Q3 + 1.5*IQR)]

        # 중복 제거
        self.df.drop_duplicates(subset=['participant_id'], inplace=True)

        print(f"✅ 클리닝 완료: {self.df.shape}")
        return self

    def explore_data(self):
        """탐색적 데이터 분석"""
        self.results['summary'] = self.df.describe()
        self.results['group_stats'] = self.df.groupby('condition').describe()
        print("✅ 탐색적 분석 완료")
        return self

    def run_statistical_tests(self):
        """통계 검정"""
        from scipy import stats

        groups = [group['dependent_var'].values
                 for name, group in self.df.groupby('condition')]

        f_stat, p_value = stats.f_oneway(*groups)
        self.results['anova'] = {'f_stat': f_stat, 'p_value': p_value}

        print(f"✅ ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        return self

    def create_visualizations(self):
        """시각화 생성"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 박스플롯
        self.df.boxplot(column='dependent_var', by='condition', ax=axes[0,0])
        axes[0,0].set_title('DV by Condition')

        # 히스토그램
        for condition in self.df['condition'].unique():
            data = self.df[self.df['condition'] == condition]['dependent_var']
            axes[0,1].hist(data, alpha=0.5, label=condition)
        axes[0,1].set_title('Distribution by Condition')
        axes[0,1].legend()

        # 평균 비교
        means = self.df.groupby('condition')['dependent_var'].mean()
        axes[1,0].bar(means.index, means.values)
        axes[1,0].set_title('Mean by Condition')

        # 산점도
        for condition in self.df['condition'].unique():
            data = self.df[self.df['condition'] == condition]
            axes[1,1].scatter(data['covariate'], data['dependent_var'],
                            label=condition, alpha=0.5)
        axes[1,1].set_title('DV vs Covariate')
        axes[1,1].legend()

        plt.tight_layout()
        plt.savefig('experiment_analysis.png', dpi=300)
        print("✅ 시각화 저장 완료")
        return self

    def generate_report(self, output_path='report.md'):
        """마크다운 보고서 생성"""
        report = f"""# 실험 분석 보고서

## 1. 데이터 개요
- 총 참가자 수: {len(self.df)}
- 실험 조건: {', '.join(self.df['condition'].unique())}

## 2. 기술 통계
{self.results['group_stats'].to_markdown()}

## 3. 통계 검정 결과
- ANOVA F-statistic: {self.results['anova']['f_stat']:.4f}
- p-value: {self.results['anova']['p_value']:.4f}
- 결론: {'유의미한 차이 있음' if self.results['anova']['p_value'] < 0.05 else '유의미한 차이 없음'}

## 4. 시각화
![실험 분석](experiment_analysis.png)

## 5. 결론
[분석 결과 요약 및 제안]
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 보고서 저장: {output_path}")
        return self

# 사용 예시
pipeline = (ExperimentAnalysisPipeline('experiment_data.csv')
            .load_data()
            .clean_data()
            .explore_data()
            .run_statistical_tests()
            .create_visualizations()
            .generate_report())
```

---

## 3. 자동화된 보고서 생성 (40분)

### Markdown 보고서

```python
def create_markdown_report(df, results, output_file='report.md'):
    """종합 분석 보고서 생성"""

    report_content = f"""# 데이터 분석 보고서

**작성일**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**분석자**: [이름]

---

## Executive Summary

이 보고서는 [실험/연구 주제]에 대한 데이터 분석 결과를 요약합니다.

### 주요 발견사항:
1. [핵심 인사이트 1]
2. [핵심 인사이트 2]
3. [핵심 인사이트 3]

---

## 1. 데이터 개요

- **총 샘플 수**: {len(df):,}
- **변수 수**: {len(df.columns)}
- **분석 기간**: {df['date'].min()} ~ {df['date'].max()}

### 데이터 구조
```
{df.info()}
```

---

## 2. 기술 통계

### 전체 통계
{df.describe().to_markdown()}

### 그룹별 통계
{df.groupby('condition')['dependent_var'].describe().to_markdown()}

---

## 3. 통계 분석 결과

### ANOVA
- F-statistic: {results['f_stat']:.4f}
- p-value: {results['p_value']:.4f}
- **결론**: {결과 해석}

### 사후 분석
[Tukey HSD 결과]

---

## 4. 시각화

### 조건별 비교
![Box Plot](boxplot.png)

### 분포 분석
![Histogram](histogram.png)

---

## 5. 결론 및 제안

### 주요 결론
1. [결론 1]
2. [결론 2]

### 실무 제안
1. [제안 1]
2. [제안 2]

### 한계점 및 향후 연구
- [한계점 1]
- [향후 연구 방향]

---

## 부록

### A. 데이터 전처리 과정
- 결측값 처리: [방법]
- 이상치 처리: [방법]

### B. 통계 가정 검정
- 정규성: [결과]
- 등분산성: [결과]
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 보고서 생성 완료: {output_file}")
```

### HTML 보고서 (더 고급)

```python
def create_html_report(df, output_file='report.html'):
    """인터랙티브 HTML 보고서"""

    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Plotly 차트들 생성
    fig1 = px.box(df, x='condition', y='dependent_var',
                  title='Dependent Variable by Condition')

    fig2 = px.histogram(df, x='dependent_var', color='condition',
                        title='Distribution of DV')

    # HTML 템플릿
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>분석 보고서</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2C3E50; }}
            h2 {{ color: #34495E; border-bottom: 2px solid #3498DB; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498DB; color: white; }}
        </style>
    </head>
    <body>
        <h1>📊 데이터 분석 보고서</h1>

        <h2>1. 데이터 개요</h2>
        <p>총 샘플 수: {len(df):,}</p>

        <h2>2. 시각화</h2>
        {fig1.to_html(full_html=False)}
        {fig2.to_html(full_html=False)}

        <h2>3. 통계 요약</h2>
        {df.describe().to_html()}

        <h2>4. 결론</h2>
        <ul>
            <li>주요 발견사항 1</li>
            <li>주요 발견사항 2</li>
        </ul>
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"✅ HTML 보고서 생성: {output_file}")
```

---

## 4. 최종 프로젝트 (20분)

### 프로젝트 주제

다음 중 하나를 선택:
1. A/B 테스트 분석 (광고, UI 등)
2. 소셜 미디어 캠페인 효과 분석
3. 뉴스 소비 패턴 분석
4. 고객 만족도 조사 분석

### 프로젝트 요구사항

**필수 포함 사항:**
1. 데이터 클리닝
2. 탐색적 데이터 분석 (EDA)
3. 통계 검정 (t-test 또는 ANOVA)
4. 최소 5개 시각화
5. 자동화된 보고서

**보고서 구조:**
```markdown
# 제목

## Executive Summary
## 1. Introduction
## 2. Methods
## 3. Results
## 4. Discussion
## 5. Conclusion
## References
```

### 평가 기준

| 항목 | 배점 |
|------|------|
| 데이터 전처리 | 20점 |
| 통계 분석 | 25점 |
| 시각화 | 25점 |
| 보고서 품질 | 20점 |
| 인사이트 | 10점 |

---

## 💡 핵심 요약

### 데이터 분석 체크리스트

- [ ] 데이터 로드 및 확인
- [ ] 결측값 처리
- [ ] 이상치 확인 및 처리
- [ ] 탐색적 데이터 분석
- [ ] 통계 검정
- [ ] 시각화
- [ ] 해석 및 보고서

### 좋은 보고서의 특징

1. **명확한 구조**: 서론-방법-결과-결론
2. **시각적**: 표와 그래프 적극 활용
3. **간결함**: 핵심만 전달
4. **실행 가능**: 구체적인 제안
5. **재현 가능**: 코드 및 데이터 제공

---

## 🤖 AI 협업 가이드

```
1. "이 데이터의 이상치를 자동으로 탐지하고 처리하는 함수 만들어줘"

2. "실험 결과를 경영진에게 보고하는 1페이지 요약 작성해줘"

3. "이 분석 파이프라인을 재사용 가능한 클래스로 리팩토링해줘"

4. "통계 결과를 비전공자도 이해할 수 있게 설명해줘"

5. "인터랙티브 대시보드를 Plotly Dash로 만들고 싶어"
```

---

## 🎉 과정 완료!

축하합니다! 5주간의 데이터 분석 여정을 완주하셨습니다!

### 배운 것들:
✅ AI 협업 코딩 (바이브 코딩)
✅ 데이터 분석 기초 (Pandas)
✅ 시각화 (Matplotlib, Seaborn)
✅ 통계 분석 (t-test, ANOVA, 회귀)
✅ 텍스트 분석 (워드클라우드, 감성 분석)
✅ 실험 데이터 처리 및 보고서 작성

### 다음 단계:
- 더 많은 프로젝트 실습
- 고급 머신러닝 탐구
- 실제 연구/업무에 적용

**계속 학습하고, AI와 함께 성장하세요! 🚀**
