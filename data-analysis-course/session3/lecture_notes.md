# Session 3: 통계 분석 기초

**수업 시간:** 2시간
**목표:** 커뮤니케이션 연구에 필요한 기본 통계 분석 마스터하기

---

## 📋 수업 목차

1. **통계의 기초** (20분)
2. **기술 통계** (25분)
3. **가설 검정 (t-test, ANOVA)** (40분)
4. **상관관계와 회귀 분석** (35분)

---

## 1. 통계의 기초 (20분)

### 왜 통계가 필요한가?

**커뮤니케이션 연구의 질문들:**
- 새 광고 캠페인이 정말 효과가 있었나?
- 남녀의 소셜 미디어 사용 패턴이 다른가?
- 연령이 뉴스 신뢰도에 영향을 미치는가?

→ **통계는 이런 질문에 과학적으로 답하는 도구!**

### 기본 개념

#### 모집단 vs 표본
- **모집단 (Population)**: 연구 대상 전체
- **표본 (Sample)**: 모집단에서 뽑은 일부
- **목표**: 표본 분석으로 모집단 추론

#### 변수 유형
1. **범주형 (Categorical)**
   - 명목형: 성별, 플랫폼, 카테고리
   - 순서형: 학년, 만족도 (매우 불만족~매우 만족)

2. **수치형 (Numerical)**
   - 이산형: 댓글 수, 공유 수
   - 연속형: 시간, 점수

#### 확률 분포
- **정규분포**: 많은 자연 현상이 따름 (종 모양)
- **중심극한정리**: 표본 크기가 충분히 크면 표본 평균은 정규분포를 따름

---

## 2. 기술 통계 (25분)

### 중심 경향성 (Central Tendency)

#### 평균 (Mean)
```python
df['likes'].mean()
```
- 장점: 모든 값 반영
- 단점: 이상치에 민감

#### 중앙값 (Median)
```python
df['likes'].median()
```
- 장점: 이상치에 강건
- 단점: 모든 값 정보 미사용

#### 최빈값 (Mode)
```python
df['platform'].mode()
```
- 가장 자주 나타나는 값

### 분산도 (Variability)

#### 분산 (Variance)
```python
df['likes'].var()
```
- 평균으로부터 얼마나 퍼져있는가?

#### 표준편차 (Standard Deviation)
```python
df['likes'].std()
```
- 분산의 제곱근
- 원래 단위로 표현

#### 사분위수 (Quartiles)
```python
df['likes'].quantile([0.25, 0.5, 0.75])
```
- Q1 (25%), Q2 (50% = 중앙값), Q3 (75%)
- IQR = Q3 - Q1

### 실전 예제
```python
import pandas as pd
import numpy as np

df = pd.read_csv('../datasets/advertising_experiment.csv')

# 기술 통계 요약
print(df.describe())

# 그룹별 통계
print(df.groupby('ad_type').agg({
    'engagement': ['mean', 'median', 'std'],
    'conversion_rate': ['mean', 'std']
}))
```

---

## 3. 가설 검정 (40분)

### 가설 검정의 논리

#### 단계
1. **귀무가설 (H0)**: 차이가 없다
2. **대립가설 (H1)**: 차이가 있다
3. **유의수준 (α)**: 보통 0.05 (5%)
4. **p-value 계산**: 귀무가설이 참일 확률
5. **결론**: p < α이면 H0 기각

#### 해석
- **p < 0.05**: 통계적으로 유의미한 차이
- **p ≥ 0.05**: 유의미한 차이 없음

⚠️ **주의**: p-value는 효과 크기가 아님!

### t-test (두 집단 비교)

#### 독립 t-test
```python
from scipy import stats

# 남성 vs 여성의 참여도 차이
male = df[df['gender'] == 'Male']['engagement']
female = df[df['gender'] == 'Female']['engagement']

t_stat, p_value = stats.ttest_ind(male, female)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ 성별에 따라 참여도에 유의미한 차이가 있습니다")
else:
    print("❌ 성별에 따른 유의미한 차이를 발견하지 못했습니다")
```

#### 대응 t-test
```python
# 동일 사람의 광고 노출 전/후 비교
before = df['engagement_before']
after = df['engagement_after']

t_stat, p_value = stats.ttest_rel(before, after)
```

### ANOVA (셋 이상 집단 비교)

```python
from scipy import stats

# 3개 이상의 광고 유형 비교
groups = [
    df[df['ad_type'] == 'Image']['engagement'],
    df[df['ad_type'] == 'Video']['engagement'],
    df[df['ad_type'] == 'Carousel']['engagement'],
    df[df['ad_type'] == 'Story']['engagement']
]

f_stat, p_value = stats.f_oneway(*groups)

print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ 광고 유형 간 유의미한 차이가 있습니다")
    print("→ 사후 분석(post-hoc test)으로 어느 그룹이 다른지 확인 필요")
```

#### 사후 분석 (Tukey HSD)
```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(df['engagement'], df['ad_type'], alpha=0.05)
print(tukey)
```

### AI 활용 팁
```
프롬프트: "t-test 결과 p-value가 0.023이 나왔어.
이게 실무적으로 어떤 의미인지 설명해줘"
```

---

## 4. 상관관계와 회귀 분석 (35분)

### 상관관계 (Correlation)

#### Pearson 상관계수
```python
# 두 변수 간 상관관계
correlation = df['likes'].corr(df['shares'])
print(f"Correlation: {correlation:.4f}")

# 전체 상관관계 행렬
corr_matrix = df[['likes', 'shares', 'comments', 'time_spent']].corr()
print(corr_matrix)

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

#### 해석
- **r = 1**: 완벽한 양의 상관관계
- **r = 0**: 상관관계 없음
- **r = -1**: 완벽한 음의 상관관계
- **|r| > 0.7**: 강한 상관관계
- **0.3 < |r| < 0.7**: 중간 상관관계
- **|r| < 0.3**: 약한 상관관계

⚠️ **상관관계 ≠ 인과관계**

### 단순 선형 회귀

```python
from scipy import stats

# 독립변수 X와 종속변수 Y
X = df['likes']
Y = df['shares']

# 회귀 분석
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

print(f"기울기 (slope): {slope:.4f}")
print(f"절편 (intercept): {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4f}")

# 회귀선 그리기
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5)
plt.plot(X, slope*X + intercept, color='red', linewidth=2,
         label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Likes')
plt.ylabel('Shares')
plt.title(f'Linear Regression (R² = {r_value**2:.4f})')
plt.legend()
plt.show()
```

#### R² (결정계수) 해석
- **R² = 0.8**: 독립변수가 종속변수의 80% 설명
- **R² = 0.5**: 50% 설명
- **R² < 0.3**: 모델 설명력 낮음

### 다중 회귀 (Multiple Regression)

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# 여러 독립변수로 예측
X = df[['likes', 'comments', 'time_spent']]
Y = df['engagement_score']

model = LinearRegression()
model.fit(X, Y)

print("회귀 계수:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

print(f"\n절편: {model.intercept_:.4f}")
print(f"R-squared: {model.score(X, Y):.4f}")

# 예측
new_data = pd.DataFrame({
    'likes': [1000],
    'comments': [50],
    'time_spent': [120]
})
prediction = model.predict(new_data)
print(f"\n예측 참여도: {prediction[0]:.2f}")
```

### 실전 예제: 광고 효과 분석

```python
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('../datasets/advertising_experiment.csv')

# 1. 기술 통계
print("=== 광고 유형별 기술 통계 ===")
print(df.groupby('ad_type')['conversion_rate'].describe())

# 2. ANOVA
groups = [group['conversion_rate'] for name, group in df.groupby('ad_type')]
f_stat, p_value = stats.f_oneway(*groups)
print(f"\nANOVA p-value: {p_value:.4f}")

# 3. 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='ad_type', y='conversion_rate', data=df)
plt.title('Conversion Rate by Ad Type')

plt.subplot(1, 2, 2)
sns.violinplot(x='ad_type', y='conversion_rate', data=df)
plt.title('Distribution of Conversion Rate')

plt.tight_layout()
plt.show()

# 4. 사후 분석
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['conversion_rate'], df['ad_type'])
print("\n=== Tukey HSD 사후 분석 ===")
print(tukey)
```

---

## 💡 핵심 요약

### 통계 분석 체크리스트

1. **데이터 탐색**
   - [ ] 기술 통계 확인
   - [ ] 분포 시각화
   - [ ] 이상치 확인

2. **가설 설정**
   - [ ] 귀무가설/대립가설 명확히
   - [ ] 유의수준 설정 (보통 0.05)

3. **적절한 검정 선택**
   - 두 집단 비교 → t-test
   - 셋 이상 비교 → ANOVA
   - 관계 분석 → 상관/회귀

4. **결과 해석**
   - [ ] p-value 확인
   - [ ] 효과 크기 고려
   - [ ] 실무적 의미 파악

### 자주 하는 실수

❌ **상관관계를 인과관계로 해석**
- 아이스크림 판매 ↑, 익사 사고 ↑
- → 아이스크림이 익사를 유발? NO! (온도가 숨은 변수)

❌ **p-value만 맹신**
- p < 0.05이지만 효과 크기가 매우 작을 수 있음

❌ **다중 비교 문제**
- 여러 번 검정하면 우연히 유의미한 결과 나올 확률 증가

---

## 🤖 AI 협업 가이드

### 유용한 프롬프트

```
1. 검정 방법 선택:
   "3개의 광고 유형과 2개의 연령대가 전환율에 미치는 영향을 분석하고 싶어.
    어떤 통계 검정을 사용해야 할까?"

2. 결과 해석:
   "ANOVA 결과 p=0.032가 나왔어. 이게 실무적으로 의미있는 건지,
    그리고 다음에 뭘 해야 하는지 알려줘"

3. 가정 확인:
   "t-test를 하기 전에 확인해야 할 가정들이 뭐야?
    그리고 Python으로 확인하는 코드를 보여줘"

4. 효과 크기:
   "통계적으로 유의미하다고 나왔는데, 효과 크기도 계산해야 한다고 들었어.
    Cohen's d를 계산하는 법 알려줘"
```

---

## 📝 다음 수업 예고

**Session 4: 텍스트 분석 & 워드클라우드**
- 텍스트 전처리
- 워드클라우드 생성
- 감성 분석 (Sentiment Analysis)
- 소셜 미디어 데이터 분석

---

**통계는 어렵지 않아요! AI와 함께라면 충분히 할 수 있습니다! 📊**
