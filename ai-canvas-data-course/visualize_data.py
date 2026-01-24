import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows의 경우 Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 읽기
df = pd.read_csv('data.csv')

# 2. 카테고리별 평균값 계산
result = df.groupby('category')['value'].mean().reset_index()

print("--- 카테고리별 평균값 ---")
print(result)

# 3. 막대그래프로 시각화
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='category', y='value', data=result, palette='viridis')

# 각 막대 끝에 평균값 라벨 추가
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

plt.title('카테고리별 평균값', fontsize=15)
plt.xlabel('카테고리', fontsize=12)
plt.ylabel('평균값', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, result['value'].max() * 1.1) # 라벨이 잘리지 않도록 y축 범위 조정

# 결과 저장
plt.savefig('category_average.png')
print("\n그래프가 'category_average.png'로 저장되었습니다.")
