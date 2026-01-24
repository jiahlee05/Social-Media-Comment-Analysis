import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 데이터 로드
try:
    df = pd.read_csv('datasets/social_media_engagement.csv')
    print("데이터 로드 성공!")
except FileNotFoundError:
    df = pd.read_csv('d:/python/AI_leture_for_Ghent_University_/dotfiles/data-analysis-course/dotfiles/ai-canvas-data-course/datasets/social_media_engagement.csv')

# 2x2 서브플롯 생성
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "1. 플랫폼별 평균 좋아요 (막대)", 
        "2. 시간대별 좋아요 트렌드 (선)",
        "3. 지표 간 상관관계 (히트맵)", 
        "4. 플랫폼별 좋아요 분포 (바이올린)"
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

# 1. 플랫폼별 평균 (막대)
platform_avg = df.groupby('platform')['likes'].mean().reset_index()
fig.add_trace(
    go.Bar(x=platform_avg['platform'], y=platform_avg['likes'], name='평균 좋아요', marker_color='RoyalBlue'),
    row=1, col=1
)

# 2. 시간 트렌드 (선)
hourly_trend = df.groupby('post_hour')['likes'].mean().reset_index()
fig.add_trace(
    go.Scatter(x=hourly_trend['post_hour'], y=hourly_trend['likes'], mode='lines+markers', name='시간대별 트렌드', line=dict(color='FireBrick')),
    row=1, col=2
)

# 3. 상관관계 (히트맵)
corr = df[['likes', 'shares', 'comments', 'post_hour']].corr()
fig.add_trace(
    go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        showscale=False
    ),
    row=2, col=1
)

# 4. 분포 (바이올린)
# Plotly Express의 바이올린을 서브플롯에 넣으려면 traces를 추출해야 함
for platform in df['platform'].unique():
    fig.add_trace(
        go.Violin(
            y=df[df['platform'] == platform]['likes'],
            name=platform,
            box_visible=True,
            meanline_visible=True
        ),
        row=2, col=2
    )

# 전체 레이아웃 설정
fig.update_layout(
    title_text="종합 분석 대시보드 (Plotly Interactive)",
    title_x=0.5,
    height=900,
    width=1100,
    showlegend=False,
    template="plotly_white",
    font=dict(family="Arial", size=12) # Plotly는 기본적으로 한글 지원이 잘 되며, 폰트 설정을 Arial 등으로 해도 브라우저에서 잘 보임
)

# 파일 저장 및 출력
output_html = 'session2_dashboard_plotly.html'
fig.write_html(output_html)
print(f"결과 시각화가 {output_html}에 저장되었습니다. 브라우저에서 확인하세요.")
