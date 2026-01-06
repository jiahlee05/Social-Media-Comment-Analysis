# Session 5: 실험 데이터 처리 & 보고서 작성
# Code Examples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

print("=" * 60)
print("Session 5: 실험 데이터 처리 & 보고서 작성")
print("=" * 60)

# ============================================
# Part 1: 데이터 클리닝
# ============================================

print("\n" + "=" * 60)
print("Part 1: 데이터 클리닝")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('../datasets/communication_experiment.csv')
print(f"\n원본 데이터: {df.shape}")

# 결측값 확인
print("\n결측값 확인:")
missing = df.isnull().sum()
print(missing[missing > 0])

# 결측값 시각화
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Pattern')
plt.tight_layout()
plt.savefig('session5_missing_data.png', dpi=300)
plt.show()

# 결측값 처리
df_clean = df.copy()
df_clean.dropna(subset=['participant_id', 'condition'], inplace=True)
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)

print(f"\n클리닝 후: {df_clean.shape}")

# 이상치 탐지
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

if 'response_time' in df_clean.columns:
    outliers, lb, ub = detect_outliers_iqr(df_clean, 'response_time')
    print(f"\n이상치 탐지 (response_time):")
    print(f"  범위: [{lb:.2f}, {ub:.2f}]")
    print(f"  이상치 개수: {len(outliers)}")

# ============================================
# Part 2: 분석 파이프라인
# ============================================

print("\n" + "=" * 60)
print("Part 2: 분석 파이프라인")
print("=" * 60)

class ExperimentAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.results = {}

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"✅ 데이터 로드: {self.df.shape}")
        return self

    def clean_data(self):
        # 결측값 제거
        self.df.dropna(subset=['participant_id', 'condition'], inplace=True)

        # 중복 제거
        before = len(self.df)
        self.df.drop_duplicates(subset=['participant_id'], inplace=True)
        after = len(self.df)
        if before != after:
            print(f"  중복 제거: {before - after}개")

        print(f"✅ 클리닝 완료: {self.df.shape}")
        return self

    def analyze(self):
        # 기술 통계
        self.results['descriptive'] = self.df.groupby('condition').describe()

        # ANOVA
        if 'persuasion_score' in self.df.columns:
            groups = [group['persuasion_score'].values
                     for name, group in self.df.groupby('condition')]
            f_stat, p_value = stats.f_oneway(*groups)
            self.results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            print(f"✅ ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

        return self

    def visualize(self, save_path='analysis_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 박스플롯
        if 'persuasion_score' in self.df.columns:
            sns.boxplot(x='condition', y='persuasion_score', data=self.df, ax=axes[0,0])
            axes[0,0].set_title('Persuasion Score by Condition', fontweight='bold')

        # 바이올린 플롯
        if 'credibility_score' in self.df.columns:
            sns.violinplot(x='condition', y='credibility_score', data=self.df, ax=axes[0,1])
            axes[0,1].set_title('Credibility Score by Condition', fontweight='bold')

        # 평균 비교
        if 'persuasion_score' in self.df.columns:
            means = self.df.groupby('condition')['persuasion_score'].mean()
            stds = self.df.groupby('condition')['persuasion_score'].std()
            axes[1,0].bar(range(len(means)), means.values, yerr=stds.values, capsize=5)
            axes[1,0].set_xticks(range(len(means)))
            axes[1,0].set_xticklabels(means.index, rotation=45)
            axes[1,0].set_title('Mean Persuasion Score by Condition', fontweight='bold')
            axes[1,0].set_ylabel('Mean Score')

        # 참가자 수
        counts = self.df['condition'].value_counts()
        axes[1,1].bar(counts.index, counts.values)
        axes[1,1].set_title('Sample Size by Condition', fontweight='bold')
        axes[1,1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 시각화 저장: {save_path}")
        plt.show()

        return self

    def generate_report(self, output_path='experiment_report.md'):
        report = f"""# 커뮤니케이션 실험 분석 보고서

**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Executive Summary

본 보고서는 커뮤니케이션 메시지 프레이밍 실험의 결과를 분석합니다.

### 주요 발견사항:
- 총 참가자 수: {len(self.df)}명
- 실험 조건: {', '.join(self.df['condition'].unique())}
- 통계적 유의성: {'있음' if self.results.get('anova', {}).get('significant', False) else '없음'}

---

## 2. 데이터 개요

### 기본 정보
- 참가자 수: {len(self.df)}
- 변수 수: {len(self.df.columns)}
- 실험 조건: {len(self.df['condition'].unique())}개

### 조건별 샘플 크기
{self.df['condition'].value_counts().to_markdown()}

---

## 3. 분석 결과

### 기술 통계
"""
        if 'persuasion_score' in self.df.columns:
            desc = self.df.groupby('condition')['persuasion_score'].describe()
            report += f"\n{desc.to_markdown()}\n"

        if 'anova' in self.results:
            report += f"""
### 통계 검정 (ANOVA)
- F-통계량: {self.results['anova']['f_statistic']:.4f}
- p-value: {self.results['anova']['p_value']:.4f}
- 결론: {'조건 간 유의미한 차이 있음 (p < 0.05)' if self.results['anova']['significant'] else '조건 간 유의미한 차이 없음'}
"""

        report += """
---

## 4. 시각화

![분석 결과](analysis_results.png)

---

## 5. 결론 및 제안

### 결론
[분석 결과를 바탕으로 한 결론]

### 실무적 제안
1. [제안 1]
2. [제안 2]
3. [제안 3]

### 한계점
- 표본 크기
- 외적 타당도
- 측정 도구

---

**보고서 끝**
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 보고서 생성: {output_path}")
        return self

# 파이프라인 실행
print("\n분석 파이프라인 실행:")
analysis = (ExperimentAnalysis('../datasets/communication_experiment.csv')
            .load_data()
            .clean_data()
            .analyze()
            .visualize()
            .generate_report())

# ============================================
# Part 3: 종합 대시보드
# ============================================

print("\n" + "=" * 60)
print("Part 3: 종합 대시보드")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 조건별 점수 분포
ax1 = fig.add_subplot(gs[0, :2])
if 'persuasion_score' in df_clean.columns:
    sns.boxplot(x='condition', y='persuasion_score', data=df_clean, ax=ax1)
    ax1.set_title('1. Persuasion Score by Condition', fontsize=13, fontweight='bold')

# 2. 통계 요약
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
if 'anova' in analysis.results:
    stats_text = f"""
통계 요약

ANOVA:
F = {analysis.results['anova']['f_statistic']:.3f}
p = {analysis.results['anova']['p_value']:.4f}

유의성:
{'✅ 유의미' if analysis.results['anova']['significant'] else '❌ 비유의미'}

샘플:
N = {len(df_clean)}
"""
    ax2.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))

# 3. 연령 분포
ax3 = fig.add_subplot(gs[1, 0])
if 'age' in df_clean.columns:
    ax3.hist(df_clean['age'], bins=20, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Count')
    ax3.set_title('2. Age Distribution')

# 4. 성별 분포
ax4 = fig.add_subplot(gs[1, 1])
if 'gender' in df_clean.columns:
    gender_counts = df_clean['gender'].value_counts()
    ax4.pie(gender_counts.values, labels=gender_counts.index,
           autopct='%1.1f%%', startangle=90)
    ax4.set_title('3. Gender Distribution')

# 5. 조건별 샘플 수
ax5 = fig.add_subplot(gs[1, 2])
condition_counts = df_clean['condition'].value_counts()
ax5.barh(condition_counts.index, condition_counts.values, color='coral')
ax5.set_xlabel('Count')
ax5.set_title('4. Sample Size by Condition')

# 6. 점수 상관관계
ax6 = fig.add_subplot(gs[2, :])
score_cols = [col for col in df_clean.columns if 'score' in col.lower()]
if len(score_cols) >= 2:
    corr = df_clean[score_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, ax=ax6, cbar_kws={'shrink': 0.8})
    ax6.set_title('5. Score Correlations')

plt.suptitle('Communication Experiment Analysis Dashboard',
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig('session5_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 종합 대시보드 완성!")

# ============================================
# 마무리
# ============================================

print("\n" + "=" * 60)
print("🎉 Session 5 완료!")
print("=" * 60)
print("""
오늘 배운 것:
✅ 데이터 클리닝 (결측값, 이상치)
✅ 분석 파이프라인 구축
✅ 자동화된 보고서 생성
✅ 종합 대시보드 작성

5주 과정 완료! 축하합니다! 🎊

이제 여러분은:
• AI와 협업하여 데이터 분석 가능
• 통계적 검정과 해석 가능
• 시각화와 보고서 작성 가능
• 실무에 바로 적용 가능

계속 학습하고 실전에 적용하세요! 🚀
""")
