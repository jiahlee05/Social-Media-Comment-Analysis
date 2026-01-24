import pandas as pd
import matplotlib
matplotlib.use('Agg') # GUI 없이 파일 저장용 백엔드 설정
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시각화 스타일 설정
sns.set_theme(style="whitegrid", palette="Blues_d")

# 1. 데이터 로드
try:
    df = pd.read_csv('datasets/social_media_engagement.csv')
    print("데이터 로드 성공!")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    # 대체 경로 시도
    df = pd.read_csv('d:/python/AI_leture_for_Ghent_University_/dotfiles/data-analysis-course/dotfiles/ai-canvas-data-course/datasets/social_media_engagement.csv')

# 2x2 subplot 설정
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('종합 분석 대시보드', fontsize=24, fontweight='bold', color='#1f4e79', y=0.95)

# 1. 플랫폼별 평균 (막대)
platform_avg = df.groupby('platform')['likes'].mean().sort_values(ascending=False)
sns.barplot(x=platform_avg.index, y=platform_avg.values, ax=axes[0, 0], palette='Blues_r')
axes[0, 0].set_title('1. 플랫폼별 평균 좋아요 수', fontsize=15)
axes[0, 0].set_xlabel('플랫폼')
axes[0, 0].set_ylabel('평균 좋아요')

# 2. 시간 트렌드 (선)
hourly_trend = df.groupby('post_hour')['likes'].mean()
sns.lineplot(x=hourly_trend.index, y=hourly_trend.values, ax=axes[0, 1], marker='o', color='#1f77b4')
axes[0, 1].set_title('2. 시간대별 좋아요 트렌드', fontsize=15)
axes[0, 1].set_xlabel('시간 (Hour)')
axes[0, 1].set_ylabel('평균 좋아요')
axes[0, 1].set_xticks(range(0, 25, 2))

# 3. 상관관계 (히트맵)
corr = df[['likes', 'shares', 'comments', 'post_hour']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('3. 지표 간 상관관계', fontsize=15)

# 4. 분포 (바이올린)
sns.violinplot(x='platform', y='likes', data=df, ax=axes[1, 1], palette='Blues')
axes[1, 1].set_title('4. 플랫폼별 좋아요 분포', fontsize=15)
axes[1, 1].set_xlabel('플랫폼')
axes[1, 1].set_ylabel('좋아요 수')

# 레이아웃 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.92])

# 결과 저장
output_path = 'session2_dashboard_result.png'
plt.savefig(output_path, dpi=300)
print(f"결과 시각화가 {output_path}에 저장되었습니다.")
