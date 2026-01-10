# Session 3: 통계 분석 기초
# Code Examples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("=" * 60)
print("Session 3: 통계 분석 기초")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('./datasets/advertising_experiment.csv')
print(f"\n데이터 로드 완료: {df.shape}")
print(df.head())

# ============================================
# Part 1: 기술 통계
# ============================================

print("\n" + "=" * 60)
print("Part 1: 기술 통계")
print("=" * 60)

# 전체 기술 통계
print("\n기본 기술 통계:")
print(df.describe())

# 그룹별 통계
print("\n광고 유형별 통계:")
ad_stats = df.groupby('ad_type').agg({
    'engagement': ['count', 'mean', 'median', 'std'],
    'conversion_rate': ['mean', 'std'],
    'cost_per_click': ['mean', 'median']
}).round(2)
print(ad_stats)

# 사분위수
print("\n참여도 사분위수:")
print(df['engagement'].quantile([0.25, 0.5, 0.75]))

# ============================================
# Part 2: t-test (두 집단 비교)
# ============================================

print("\n" + "=" * 60)
print("Part 2: t-test")
print("=" * 60)

# 예제: 남성 vs 여성 참여도 비교
if 'gender' in df.columns:
    male = df[df['gender'] == 'Male']['engagement']
    female = df[df['gender'] == 'Female']['engagement']

    t_stat, p_value = stats.ttest_ind(male, female)

    print(f"\n남성 평균 참여도: {male.mean():.2f}")
    print(f"여성 평균 참여도: {female.mean():.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✅ 성별에 따라 참여도에 유의미한 차이가 있습니다")
    else:
        print("❌ 성별에 따른 유의미한 차이를 발견하지 못했습니다")

    # 시각화
    plt.figure(figsize=(10, 6))
    data_to_plot = [male, female]
    plt.boxplot(data_to_plot, labels=['Male', 'Female'])
    plt.ylabel('Engagement')
    plt.title(f't-test: Male vs Female (p={p_value:.4f})')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('session3_ttest.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# Part 3: ANOVA (여러 집단 비교)
# ============================================

print("\n" + "=" * 60)
print("Part 3: ANOVA")
print("=" * 60)

# 광고 유형별 전환율 비교
groups = [group['conversion_rate'].values
          for name, group in df.groupby('ad_type')]

f_stat, p_value = stats.f_oneway(*groups)

print(f"\nF-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ 광고 유형 간 유의미한 차이가 있습니다")
else:
    print("❌ 광고 유형 간 유의미한 차이를 발견하지 못했습니다")

# 각 그룹 평균
print("\n각 광고 유형별 평균 전환율:")
print(df.groupby('ad_type')['conversion_rate'].mean().sort_values(ascending=False))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(x='ad_type', y='conversion_rate', data=df, ax=axes[0])
axes[0].set_title(f'ANOVA: Conversion Rate by Ad Type (p={p_value:.4f})')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.violinplot(x='ad_type', y='conversion_rate', data=df, ax=axes[1])
axes[1].set_title('Distribution of Conversion Rate')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('session3_anova.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Part 4: 사후 분석 (Post-hoc Test)
# ============================================

if p_value < 0.05:
    print("\n" + "=" * 60)
    print("Part 4: Tukey HSD 사후 분석")
    print("=" * 60)

    tukey = pairwise_tukeyhsd(df['conversion_rate'], df['ad_type'], alpha=0.05)
    print(tukey)

# ============================================
# Part 5: 상관관계 분석
# ============================================

print("\n" + "=" * 60)
print("Part 5: 상관관계 분석")
print("=" * 60)

# 상관계수 행렬
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

print("\n상관계수 행렬:")
print(corr_matrix.round(3))

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=1)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('session3_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 특정 변수 간 상관관계
if 'engagement' in df.columns and 'conversion_rate' in df.columns:
    r, p = stats.pearsonr(df['engagement'], df['conversion_rate'])
    print(f"\n참여도 vs 전환율:")
    print(f"  상관계수 (r): {r:.4f}")
    print(f"  p-value: {p:.4f}")

    if abs(r) > 0.7:
        strength = "강한"
    elif abs(r) > 0.3:
        strength = "중간"
    else:
        strength = "약한"

    print(f"  → {strength} {'양의' if r > 0 else '음의'} 상관관계")

# ============================================
# Part 6: 단순 선형 회귀
# ============================================

print("\n" + "=" * 60)
print("Part 6: 단순 선형 회귀")
print("=" * 60)

if 'engagement' in df.columns and 'conversion_rate' in df.columns:
    X = df['engagement'].values
    Y = df['conversion_rate'].values

    # NaN 제거
    mask = ~np.isnan(X) & ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    print(f"\n회귀 방정식: y = {slope:.4f}x + {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"p-value: {p_value:.4f}")

    # 해석
    print(f"\n해석: 참여도가 1 증가하면 전환율은 평균적으로 {slope:.4f} 증가")
    print(f"모델은 전환율 변동의 {r_value**2*100:.1f}%를 설명")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5, s=50)
    plt.plot(X, slope*X + intercept, color='red', linewidth=2,
             label=f'y = {slope:.4f}x + {intercept:.4f}')
    plt.xlabel('Engagement', fontsize=12)
    plt.ylabel('Conversion Rate', fontsize=12)
    plt.title(f'Linear Regression (R² = {r_value**2:.4f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('session3_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# Part 7: 다중 회귀
# ============================================

print("\n" + "=" * 60)
print("Part 7: 다중 회귀 분석")
print("=" * 60)

from sklearn.linear_model import LinearRegression

# 독립변수 선택
feature_cols = ['engagement', 'cost_per_click']
if all(col in df.columns for col in feature_cols):
    X = df[feature_cols].dropna()
    Y = df.loc[X.index, 'conversion_rate']

    model = LinearRegression()
    model.fit(X, Y)

    print("\n회귀 계수:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  절편: {model.intercept_:.4f}")

    r_squared = model.score(X, Y)
    print(f"\nR-squared: {r_squared:.4f}")
    print(f"모델은 전환율 변동의 {r_squared*100:.1f}%를 설명")

    # 예측 예제
    print("\n예측 예제:")
    sample_data = pd.DataFrame({
        'engagement': [500],
        'cost_per_click': [1.5]
    })
    prediction = model.predict(sample_data)
    print(f"  입력: engagement=500, cost_per_click=1.5")
    print(f"  예측 전환율: {prediction[0]:.4f}")

# ============================================
# Part 8: 종합 실습 - 광고 효과 분석
# ============================================

print("\n" + "=" * 60)
print("Part 8: 종합 실습 - 광고 효과 분석")
print("=" * 60)

# 대시보드
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 광고 유형별 평균 전환율
ax1 = fig.add_subplot(gs[0, :2])
avg_conversion = df.groupby('ad_type')['conversion_rate'].mean().sort_values(ascending=False)
bars = ax1.bar(range(len(avg_conversion)), avg_conversion.values,
               color='skyblue', edgecolor='black')
ax1.set_xticks(range(len(avg_conversion)))
ax1.set_xticklabels(avg_conversion.index, rotation=45, ha='right')
ax1.set_ylabel('Conversion Rate')
ax1.set_title('Average Conversion Rate by Ad Type', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 2. 통계 정보 텍스트
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
📊 통계 요약

ANOVA 결과:
F = {f_stat:.2f}
p = {p_value:.4f}
{'✅ 유의미' if p_value < 0.05 else '❌ 유의미하지 않음'}

최고 성과:
{avg_conversion.index[0]}
({avg_conversion.values[0]:.2%})

표본 크기: {len(df)}
"""
ax2.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. 참여도 vs 전환율 산점도
ax3 = fig.add_subplot(gs[1, :])
for ad_type in df['ad_type'].unique():
    mask = df['ad_type'] == ad_type
    ax3.scatter(df[mask]['engagement'], df[mask]['conversion_rate'],
                label=ad_type, alpha=0.6, s=50)
ax3.set_xlabel('Engagement')
ax3.set_ylabel('Conversion Rate')
ax3.set_title('Engagement vs Conversion Rate by Ad Type', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4-6. 박스플롯
ax4 = fig.add_subplot(gs[2, 0])
sns.boxplot(y='conversion_rate', data=df, ax=ax4, color='lightblue')
ax4.set_title('Overall Distribution')

ax5 = fig.add_subplot(gs[2, 1])
sns.boxplot(x='ad_type', y='engagement', data=df, ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
ax5.set_title('Engagement by Ad Type')

ax6 = fig.add_subplot(gs[2, 2])
if 'age_group' in df.columns:
    sns.boxplot(x='age_group', y='conversion_rate', data=df, ax=ax6)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.set_title('Conversion by Age Group')

plt.suptitle('Advertising Campaign Statistical Analysis',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('session3_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# AI 프롬프트 예제
# ============================================

print("\n" + "=" * 60)
print("🤖 AI 협업 프롬프트 예제")
print("=" * 60)

print("""
다음 프롬프트를 AI에게 시도해보세요:

1. "p-value가 0.03인데 이게 실무적으로 의미있는 건지 어떻게 판단해?"

2. "정규성 가정을 확인하는 Shapiro-Wilk test 코드를 짜줘"

3. "t-test와 ANOVA의 차이를 예제와 함께 설명해줘"

4. "이 회귀 분석 결과를 비전공자에게 설명한다면 어떻게 말해야 할까?"

5. "효과 크기(effect size)를 계산하는 Cohen's d 코드를 보여줘"
""")

print("\n✅ Session 3 완료!")
