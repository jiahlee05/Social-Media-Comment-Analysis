# Session 2: 데이터 시각화 마스터하기

**수업 시간:** 2시간
**목표:** 효과적인 데이터 시각화로 스토리 전달하기

---

## 📋 수업 목차

1. **데이터 시각화의 원칙** (20분)
2. **Matplotlib 기초** (30분)
3. **Seaborn으로 고급 시각화** (40분)
4. **커뮤니케이션을 위한 비주얼 스토리텔링** (30분)

---

## 1. 데이터 시각화의 원칙 (20분)

### 왜 시각화가 중요한가?

**"A picture is worth a thousand words"**

- 인간의 뇌는 시각 정보를 빠르게 처리 (텍스트보다 60,000배 빠름)
- 패턴과 트렌드를 즉시 파악
- 복잡한 데이터를 단순하게 전달
- 의사결정 지원

### 좋은 시각화 vs 나쁜 시각화

#### ✅ 좋은 시각화의 특징
1. **명확한 목적**: 무엇을 보여주고 싶은가?
2. **적절한 차트 선택**: 데이터 특성에 맞는 차트
3. **간결함**: 불필요한 요소 제거
4. **정확성**: 데이터 왜곡 없음
5. **접근성**: 색맹을 고려한 색상 선택

#### ❌ 피해야 할 것들
- 3D 차트 (왜곡 발생)
- 너무 많은 색상
- 의미 없는 장식
- 축 범위 조작
- 비교 불가능한 차트

### 차트 타입 선택 가이드

| 목적 | 추천 차트 |
|------|-----------|
| 비교 | 막대그래프, 그룹 막대그래프 |
| 시간 추이 | 선 그래프, 영역 그래프 |
| 분포 | 히스토그램, 박스플롯 |
| 관계 | 산점도, 버블 차트 |
| 구성 비율 | 파이 차트 (5개 이하), 도넛 차트 |
| 상관관계 | 히트맵, 산점도 행렬 |

---

## 2. Matplotlib 기초 (30분)

### Matplotlib 구조 이해

```python
import matplotlib.pyplot as plt

# Figure와 Axes
fig, ax = plt.subplots()

# Figure: 전체 캔버스
# Axes: 실제 플롯이 그려지는 영역
```

### 기본 플롯 만들기

#### 1. 선 그래프
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='blue', label='sin(x)')
plt.title('Sine Wave', fontsize=16, fontweight='bold')
plt.xlabel('X axis', fontsize=12)
plt.ylabel('Y axis', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 2. 막대그래프
```python
platforms = ['Instagram', 'Twitter', 'Facebook', 'TikTok']
likes = [850, 620, 730, 920]

plt.figure(figsize=(10, 6))
plt.bar(platforms, likes, color='skyblue', edgecolor='black')
plt.title('Average Likes by Platform')
plt.ylabel('Average Likes')
plt.show()
```

#### 3. 산점도
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['likes'], df['shares'], alpha=0.5)
plt.xlabel('Likes')
plt.ylabel('Shares')
plt.title('Likes vs Shares')
plt.show()
```

### 커스터마이징 핵심

```python
# 색상
colors = ['red', '#FF5733', (0.1, 0.2, 0.5)]

# 마커
markers = ['o', 's', '^', 'D', '*']

# 선 스타일
linestyles = ['-', '--', '-.', ':']

# 크기
figsize = (12, 8)
fontsize = 14
```

### AI 활용 팁
```
프롬프트: "이 데이터를 시각화하는데 어떤 차트가 가장 적합할까?
그리고 matplotlib 코드 예제를 보여줘"
```

---

## 3. Seaborn으로 고급 시각화 (40분)

### Seaborn이란?
- Matplotlib 기반의 고수준 시각화 라이브러리
- 더 아름다운 기본 스타일
- 통계적 시각화에 특화
- 적은 코드로 복잡한 시각화

### 핵심 함수들

#### 1. 분포 시각화
```python
import seaborn as sns

# 히스토그램 + KDE
sns.histplot(df['likes'], kde=True)
plt.title('Distribution of Likes')
plt.show()

# 박스플롯 (이상치 확인)
sns.boxplot(x='platform', y='likes', data=df)
plt.title('Likes Distribution by Platform')
plt.show()

# 바이올린 플롯 (분포 + 박스플롯)
sns.violinplot(x='platform', y='likes', data=df)
plt.show()
```

#### 2. 관계 시각화
```python
# 산점도 + 회귀선
sns.regplot(x='likes', y='shares', data=df)
plt.title('Likes vs Shares with Regression Line')
plt.show()

# 산점도 행렬
sns.pairplot(df[['likes', 'shares', 'comments', 'platform']],
             hue='platform')
plt.show()
```

#### 3. 범주형 데이터 시각화
```python
# 막대그래프 (평균 자동 계산)
sns.barplot(x='platform', y='likes', data=df, ci=95)
plt.title('Average Likes by Platform (95% CI)')
plt.show()

# 포인트 플롯 (트렌드)
sns.pointplot(x='age_group', y='likes', hue='platform', data=df)
plt.title('Likes by Age Group and Platform')
plt.show()

# 카운트 플롯
sns.countplot(x='platform', data=df)
plt.title('Number of Posts by Platform')
plt.show()
```

#### 4. 상관관계 히트맵
```python
# 상관계수 계산
correlation = df[['likes', 'shares', 'comments']].corr()

# 히트맵
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm',
            center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
```

### Seaborn 스타일
```python
# 스타일 옵션
sns.set_style('whitegrid')  # whitegrid, darkgrid, white, dark, ticks

# 컨텍스트 (크기)
sns.set_context('talk')  # paper, notebook, talk, poster

# 색상 팔레트
sns.set_palette('husl')  # deep, muted, bright, pastel, dark, colorblind
```

---

## 4. 커뮤니케이션을 위한 비주얼 스토리텔링 (30분)

### 데이터 스토리의 구조

1. **Hook (관심 유발)**: 흥미로운 발견
2. **Context (맥락)**: 왜 중요한가?
3. **Evidence (증거)**: 데이터로 보여주기
4. **Conclusion (결론)**: 행동 제안

### 실전 예제: 소셜 미디어 전략 보고서

#### Step 1: 핵심 메시지 정하기
```
"Instagram이 다른 플랫폼보다 2배 높은 참여도를 보이며,
특히 20-29세 연령대에서 가장 효과적"
```

#### Step 2: 스토리보드
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 플랫폼별 비교
sns.barplot(x='platform', y='likes', data=df, ax=axes[0, 0])
axes[0, 0].set_title('1. Platform Performance Comparison',
                     fontsize=14, fontweight='bold')

# 2. 연령대별 패턴
sns.boxplot(x='age_group', y='likes', data=df, ax=axes[0, 1])
axes[0, 1].set_title('2. Engagement by Age Group',
                     fontsize=14, fontweight='bold')

# 3. 플랫폼-연령 교차 분석
sns.pointplot(x='age_group', y='likes', hue='platform',
              data=df, ax=axes[1, 0])
axes[1, 0].set_title('3. Platform Preferences by Age',
                     fontsize=14, fontweight='bold')

# 4. 시간대별 트렌드
hourly = df.groupby('post_hour')['likes'].mean()
axes[1, 1].plot(hourly.index, hourly.values, marker='o', linewidth=2)
axes[1, 1].set_title('4. Optimal Posting Time',
                     fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Average Likes')

plt.tight_layout()
plt.savefig('social_media_strategy_report.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 주석과 강조 추가
```python
plt.figure(figsize=(12, 6))
bars = plt.bar(platforms, likes, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])

# 최고값 강조
max_idx = np.argmax(likes)
bars[max_idx].set_color('#FF1744')
bars[max_idx].set_edgecolor('black')
bars[max_idx].set_linewidth(3)

# 주석 추가
plt.annotate('Best Performance!',
             xy=(max_idx, likes[max_idx]),
             xytext=(max_idx, likes[max_idx] + 100),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, fontweight='bold', color='red')

plt.title('Platform Performance Analysis', fontsize=16, fontweight='bold')
plt.ylabel('Average Likes', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 색상 전략
```python
# 색맹 친화적 팔레트
colorblind_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']

# 강조 색상
highlight_color = '#FF1744'
neutral_colors = ['#CCCCCC'] * len(platforms)
neutral_colors[max_idx] = highlight_color
```

---

## 💡 핵심 요약

### 꼭 기억할 것
1. **목적 우선**: 무엇을 전달할지 먼저 정하기
2. **적절한 차트**: 데이터 특성에 맞는 차트 선택
3. **간결함**: Less is more
4. **스토리**: 데이터로 이야기 만들기

### Matplotlib vs Seaborn

| 특징 | Matplotlib | Seaborn |
|------|------------|---------|
| 자유도 | 높음 | 중간 |
| 코드 길이 | 길음 | 짧음 |
| 통계 기능 | 제한적 | 풍부 |
| 기본 스타일 | 기본적 | 세련됨 |
| 사용 케이스 | 커스텀 플롯 | 통계 분석 |

### AI 활용 체크리스트
```
✅ "이 차트를 더 명확하게 만들려면?"
✅ "색상 조합을 개선해줘"
✅ "이 데이터의 핵심 메시지는 무엇일까?"
✅ "차트에 주석을 추가하는 코드를 짜줘"
```

---

## 📝 과제 미리보기

다음 주까지:
1. **뉴스 소비 패턴 데이터** 시각화
2. **최소 5개의 다양한 차트** 생성
3. **스토리보드** 형식으로 발표 자료 만들기

---

## 🔗 추가 학습 자료

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Data Visualization Best Practices](https://www.storytellingwithdata.com/)
- [ColorBrewer (색상 선택)](https://colorbrewer2.org/)

---

**다음 시간: 통계 분석 기초! 📊**
