# Session 2: 데이터 시각화 마스터하기
# Code Examples - Jupyter Notebook으로 실행하세요

# ============================================
# Part 1: 환경 설정
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 스타일 설정
sns.set_style('whitegrid')
sns.set_context('notebook')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("✅ 라이브러리 임포트 완료!")

# 데이터 로드
df = pd.read_csv('./datasets/news_consumption.csv')
print(f"📊 데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}열")

# ============================================
# Part 2: Matplotlib 기초
# ============================================

print("\n" + "=" * 50)
print("📈 Part 2: Matplotlib 기본 차트")
print("=" * 50)

# 예제 2-1: 선 그래프
plt.figure(figsize=(12, 6))

# 일별 뉴스 소비 트렌드 (가상 데이터)
days = np.arange(1, 31)
consumption = 50 + 10 * np.sin(days / 5) + np.random.normal(0, 3, 30)

plt.plot(days, consumption, marker='o', linewidth=2, markersize=6,
         color='#2E86AB', label='Daily News Consumption')

# 이동 평균 추가
window = 7
moving_avg = pd.Series(consumption).rolling(window=window).mean()
plt.plot(days, moving_avg, linewidth=3, color='#A23B72',
         linestyle='--', label=f'{window}-day Moving Average')

plt.title('News Consumption Trend (30 Days)', fontsize=16, fontweight='bold')
plt.xlabel('Day', fontsize=12)
plt.ylabel('Articles Read', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('session2_line_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 선 그래프 완성")

# 예제 2-2: 막대그래프
print("\n막대그래프 생성 중...")

# 뉴스 카테고리별 소비
categories = df.groupby('category')['articles_read'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
bars = plt.bar(range(len(categories)), categories.values,
               color=colors, edgecolor='black', linewidth=1.5)

# 값 표시
for i, (bar, value) in enumerate(zip(bars, categories.values)):
    plt.text(i, value + 1, f'{value:.1f}',
             ha='center', va='bottom', fontweight='bold')

plt.xticks(range(len(categories)), categories.index, rotation=45, ha='right')
plt.title('Average Articles Read by Category', fontsize=16, fontweight='bold')
plt.ylabel('Average Articles Read', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('session2_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 막대그래프 완성")

# 예제 2-3: 산점도
print("\n산점도 생성 중...")

plt.figure(figsize=(10, 8))

# 시간 vs 참여도
scatter = plt.scatter(df['time_spent'], df['engagement_score'],
                     c=df['age'], cmap='coolwarm', s=100,
                     alpha=0.6, edgecolors='black', linewidth=0.5)

plt.colorbar(scatter, label='Age')
plt.xlabel('Time Spent (minutes)', fontsize=12)
plt.ylabel('Engagement Score', fontsize=12)
plt.title('Reading Time vs Engagement Score', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('session2_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 산점도 완성")

# 예제 2-4: 히스토그램
print("\n히스토그램 생성 중...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: 단순 히스토그램
axes[0].hist(df['time_spent'], bins=30, color='skyblue',
             edgecolor='black', alpha=0.7)
axes[0].axvline(df['time_spent'].mean(), color='red',
                linestyle='--', linewidth=2,
                label=f"Mean: {df['time_spent'].mean():.1f} min")
axes[0].axvline(df['time_spent'].median(), color='green',
                linestyle='--', linewidth=2,
                label=f"Median: {df['time_spent'].median():.1f} min")
axes[0].set_xlabel('Time Spent (minutes)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Reading Time', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 오른쪽: 누적 히스토그램
axes[1].hist(df['time_spent'], bins=30, cumulative=True,
             color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Time Spent (minutes)', fontsize=12)
axes[1].set_ylabel('Cumulative Frequency', fontsize=12)
axes[1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('session2_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 히스토그램 완성")

# ============================================
# Part 3: Seaborn 고급 시각화
# ============================================

print("\n" + "=" * 50)
print("🎨 Part 3: Seaborn 고급 시각화")
print("=" * 50)

# 예제 3-1: 박스플롯 & 바이올린 플롯
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 박스플롯
sns.boxplot(x='device', y='time_spent', data=df, ax=axes[0],
            palette='Set2')
axes[0].set_title('Reading Time by Device (Boxplot)',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Device', fontsize=12)
axes[0].set_ylabel('Time Spent (minutes)', fontsize=12)

# 바이올린 플롯
sns.violinplot(x='device', y='time_spent', data=df, ax=axes[1],
               palette='Set3')
axes[1].set_title('Reading Time by Device (Violin Plot)',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Device', fontsize=12)
axes[1].set_ylabel('Time Spent (minutes)', fontsize=12)

plt.tight_layout()
plt.savefig('session2_boxplot_violin.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 박스플롯 & 바이올린 플롯 완성")

# 예제 3-2: 카테고리별 막대그래프 (CI 포함)
plt.figure(figsize=(12, 6))

sns.barplot(x='category', y='engagement_score', data=df,
            palette='rocket', ci=95, capsize=0.1)

plt.title('Engagement Score by News Category (95% CI)',
          fontsize=16, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Engagement Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('session2_barplot_ci.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 신뢰구간 막대그래프 완성")

# 예제 3-3: 포인트 플롯 (트렌드 비교)
plt.figure(figsize=(12, 6))

sns.pointplot(x='age_group', y='time_spent', hue='device',
              data=df, markers=['o', 's', '^'], linestyles=['-', '--', '-.'])

plt.title('Reading Time by Age Group and Device',
          fontsize=16, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Average Time Spent (minutes)', fontsize=12)
plt.legend(title='Device', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('session2_pointplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 포인트 플롯 완성")

# 예제 3-4: 히트맵 (상관관계)
print("\n상관관계 히트맵 생성 중...")

# 수치형 변수만 선택
numeric_cols = ['time_spent', 'articles_read', 'engagement_score', 'age']
correlation = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=1, cbar_kws={"shrink": 0.8})

plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('session2_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 히트맵 완성")

# 예제 3-5: 페어플롯 (산점도 행렬)
print("\n페어플롯 생성 중 (시간이 걸릴 수 있습니다)...")

pairplot = sns.pairplot(df[numeric_cols + ['device']],
                        hue='device', palette='husl',
                        diag_kind='kde', plot_kws={'alpha': 0.6})

pairplot.fig.suptitle('Pairwise Relationships', y=1.02,
                      fontsize=16, fontweight='bold')
plt.savefig('session2_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 페어플롯 완성")

# 예제 3-6: 회귀 플롯
plt.figure(figsize=(10, 6))

sns.regplot(x='articles_read', y='engagement_score', data=df,
            scatter_kws={'alpha': 0.5, 's': 50},
            line_kws={'color': 'red', 'linewidth': 2})

plt.title('Articles Read vs Engagement Score (with Regression Line)',
          fontsize=16, fontweight='bold')
plt.xlabel('Articles Read', fontsize=12)
plt.ylabel('Engagement Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('session2_regplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 회귀 플롯 완성")

# ============================================
# Part 4: 복합 시각화 - 스토리보드
# ============================================

print("\n" + "=" * 50)
print("📊 Part 4: 스토리보드 - 뉴스 소비 패턴 분석")
print("=" * 50)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 카테고리별 인기도
ax1 = fig.add_subplot(gs[0, :2])
category_counts = df['category'].value_counts()
ax1.barh(category_counts.index, category_counts.values, color='skyblue', edgecolor='black')
ax1.set_xlabel('Number of Readers', fontsize=11)
ax1.set_title('1. Most Popular News Categories', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. 디바이스별 분포
ax2 = fig.add_subplot(gs[0, 2])
device_counts = df['device'].value_counts()
colors_pie = ['#FF9999', '#66B2FF', '#99FF99']
ax2.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=90)
ax2.set_title('2. Device Distribution', fontsize=13, fontweight='bold')

# 3. 연령대별 참여도
ax3 = fig.add_subplot(gs[1, :])
sns.boxplot(x='age_group', y='engagement_score', hue='device',
            data=df, ax=ax3, palette='Set2')
ax3.set_title('3. Engagement Score by Age Group and Device',
              fontsize=13, fontweight='bold')
ax3.set_xlabel('Age Group', fontsize=11)
ax3.set_ylabel('Engagement Score', fontsize=11)
ax3.legend(title='Device', fontsize=9)

# 4. 독서 시간 분포
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(df['time_spent'], bins=20, color='coral', alpha=0.7, edgecolor='black')
ax4.axvline(df['time_spent'].mean(), color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Time Spent (min)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('4. Reading Time Distribution', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. 시간-참여도 관계
ax5 = fig.add_subplot(gs[2, 1])
scatter = ax5.scatter(df['time_spent'], df['engagement_score'],
                      c=df['age'], cmap='viridis', alpha=0.6, s=30)
ax5.set_xlabel('Time Spent (min)', fontsize=11)
ax5.set_ylabel('Engagement Score', fontsize=11)
ax5.set_title('5. Time vs Engagement', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax5, label='Age')
ax5.grid(True, alpha=0.3)

# 6. 주요 인사이트
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
insights_text = """
📌 Key Insights:

• Most popular: Politics
  and Technology news

• Mobile dominates with
  55% of readers

• 30-39 age group shows
  highest engagement

• Avg reading time:
  {:.1f} minutes

• Strong correlation
  between time spent
  and engagement
""".format(df['time_spent'].mean())

ax6.text(0.1, 0.9, insights_text, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('News Consumption Pattern Analysis Dashboard',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('session2_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 스토리보드 대시보드 완성!")

# ============================================
# Part 5: 주석과 강조
# ============================================

print("\n" + "=" * 50)
print("✏️ Part 5: 주석과 강조 기법")
print("=" * 50)

# 중요한 데이터 포인트 강조
plt.figure(figsize=(12, 6))

category_avg = df.groupby('category')['engagement_score'].mean().sort_values(ascending=False)
colors_bars = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(category_avg))]

bars = plt.bar(range(len(category_avg)), category_avg.values,
               color=colors_bars, edgecolor='black', linewidth=1.5)

# 최고값 주석
max_idx = 0
max_value = category_avg.values[0]
plt.annotate(f'Highest!\n{max_value:.2f}',
             xy=(max_idx, max_value),
             xytext=(max_idx + 0.5, max_value + 0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, fontweight='bold', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.xticks(range(len(category_avg)), category_avg.index, rotation=45, ha='right')
plt.title('Engagement Score by Category (with Annotations)',
          fontsize=16, fontweight='bold')
plt.ylabel('Engagement Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('session2_annotations.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 주석 추가 차트 완성")

# ============================================
# Part 6: AI 협업 실습
# ============================================

print("\n" + "=" * 50)
print("🤖 Part 6: AI 협업 프롬프트")
print("=" * 50)

print("""
다음 프롬프트를 AI에게 시도해보세요:

1. 차트 선택 조언:
   "연령대별 뉴스 카테고리 선호도를 비교하고 싶어.
    어떤 차트가 가장 적합할까?"

2. 코드 개선:
   "이 막대그래프를 더 전문적으로 보이게 만들고 싶어.
    색상, 레이블, 그리드를 개선하는 코드를 짜줘"

3. 색상 팔레트:
   "5개 카테고리를 구분하기 좋은 색맹 친화적인 색상 조합을 추천해줘"

4. 스토리텔링:
   "이 데이터 분석 결과를 3장의 슬라이드로 만든다면,
    어떤 차트를 어떤 순서로 배치해야 할까?"

5. 문제 해결:
   "범례가 차트를 가려서 안 보여. 어떻게 해결할까?"

6. 해석 도움:
   "이 히트맵에서 어떤 변수들이 강한 상관관계를 보이는지 설명해줘"
""")

# ============================================
# 마무리
# ============================================

print("\n" + "=" * 50)
print("🎉 Session 2 완료!")
print("=" * 50)
print("""
오늘 배운 것:
✅ Matplotlib 기본 차트 (선, 막대, 산점도, 히스토그램)
✅ Seaborn 고급 시각화 (박스플롯, 히트맵, 페어플롯)
✅ 복합 대시보드 구성
✅ 주석과 강조 기법
✅ 효과적인 비주얼 스토리텔링

다음 시간:
📊 통계 분석 기초
📈 가설 검정과 t-test
🔍 회귀 분석 입문

과제를 잊지 마세요! 🚀
""")
