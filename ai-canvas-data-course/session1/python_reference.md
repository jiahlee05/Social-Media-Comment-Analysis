# Python 기초 참고자료
## "이런게 있구나" 수준으로만 알아두세요!

**중요:** 이 자료는 **참고용**입니다. 외울 필요 없어요!
AI 캔버스가 이미 다 알고 있으니, 여러분은 **무엇을 분석할지**만 생각하면 됩니다.

---

## 🐍 Python이란?

- 데이터 분석에 가장 많이 쓰이는 프로그래밍 언어
- 문법이 영어와 비슷해서 읽기 쉬움
- 하지만... **여러분이 직접 쓸 일은 거의 없어요!** AI가 대신 써주니까요.

---

## 📦 주요 라이브러리 (이름만 알아두세요)

### 1. Pandas - 데이터 처리
```python
import pandas as pd  # "pandas를 pd라는 이름으로 쓸게"
```

**하는 일:**
- 엑셀 같은 표 형태 데이터 다루기
- CSV 파일 읽기/쓰기
- 데이터 정제, 계산, 그룹화

**여러분이 할 일:**
```
AI에게: "pandas로 이 CSV 읽어서 평균 계산해줘"
```

---

### 2. Matplotlib - 기본 시각화
```python
import matplotlib.pyplot as plt
```

**하는 일:**
- 막대그래프, 선 그래프, 산점도 등 기본 차트

**여러분이 할 일:**
```
AI에게: "matplotlib으로 막대그래프 그려줘"
```

---

### 3. Seaborn - 예쁜 시각화
```python
import seaborn as sns
```

**하는 일:**
- Matplotlib보다 더 예쁜 차트
- 통계 시각화 특화

**여러분이 할 일:**
```
AI에게: "seaborn으로 전문적인 차트 만들어줘"
```

---

### 4. Plotly - 인터랙티브 차트
```python
import plotly.express as px
```

**하는 일:**
- 마우스 오버, 줌, 애니메이션 등
- 웹에서 바로 사용 가능

**여러분이 할 일:**
```
AI에게: "plotly로 인터랙티브 차트 만들어줘"
```

---

## 💡 코드 읽는 법 (최소한만)

### 변수 (이름 붙이기)
```python
name = "Claude"  # name이라는 상자에 "Claude" 저장
age = 1          # age라는 상자에 1 저장
```

### 리스트 (여러 개 묶기)
```python
platforms = ["Instagram", "Twitter", "Facebook"]
# → Instagram, Twitter, Facebook을 하나로 묶음
```

### 함수 (작업 시키기)
```python
result = df.mean()  # df에게 "평균 계산해!"라고 명령
```

### 점(.) 의미
```python
df.groupby('platform').mean()
# → df야, platform으로 그룹화하고, 평균 내!
# 점은 "에게 ~해!"라는 뜻
```

---

## 📊 자주 보는 Pandas 코드 (읽기만)

### CSV 파일 읽기
```python
df = pd.read_csv('data.csv')
# → data.csv 파일을 df라는 이름으로 불러오기
```

### 처음 몇 행 보기
```python
df.head()
# → 앞쪽 5개 행만 보여줘
```

### 평균 계산
```python
df['likes'].mean()
# → likes 컬럼의 평균
```

### 그룹별 계산
```python
df.groupby('platform')['likes'].mean()
# → platform별로 묶어서 likes의 평균
```

---

## 📈 자주 보는 시각화 코드 (읽기만)

### 막대그래프
```python
plt.bar(x, y)           # 막대 그리기
plt.title('제목')       # 제목
plt.xlabel('X축 이름')  # X축
plt.ylabel('Y축 이름')  # Y축
plt.show()              # 보여주기
```

### 선 그래프
```python
plt.plot(x, y, marker='o')  # 선 + 마커
```

### 산점도
```python
plt.scatter(x, y)  # 점들 찍기
```

---

## 🎨 시각화 스타일 코드 (읽기만)

### 색상 지정
```python
color='blue'          # 파란색
color='#2E86AB'       # 색상 코드 (HEX)
color='skyblue'       # 색상 이름
```

### 크기 지정
```python
figsize=(12, 6)       # 가로 12, 세로 6 인치
fontsize=16           # 폰트 크기 16
```

### 레이블
```python
plt.title('제목', fontsize=16, fontweight='bold')
# → 제목, 크기 16, 굵게
```

---

## 🔢 기본 연산

```python
# 계산
sum = 1 + 2          # 더하기
product = 3 * 4      # 곱하기
average = sum / 2    # 나누기

# 비교
is_big = likes > 100  # likes가 100보다 큰가?

# 조건 필터링
high_engagement = df[df['likes'] > 100]
# → likes가 100 이상인 것만 골라내기
```

---

## 🤔 이 정도면 충분해요!

### 기억하세요:

1. **코드를 외울 필요 없어요**
   - AI가 다 알고 있어요
   - 여러분은 "무엇을, 어떻게"만 말하면 됨

2. **에러 나면 AI에게 물어보세요**
   ```
   "이 에러가 뭐야? 어떻게 고쳐?"
   [에러 메시지 복사-붙여넣기]
   ```

3. **코드 읽는 연습만 하세요**
   - AI가 만든 코드 이해하기
   - "이 줄은 무슨 뜻이야?" 질문하기

4. **궁금하면 AI에게**
   ```
   "이 코드 라인별로 설명해줘"
   "왜 이렇게 하는 거야?"
   "다른 방법은 없어?"
   ```

---

## 📚 더 알고 싶다면?

### Python 기초 (선택사항)
- [Python 공식 튜토리얼](https://docs.python.org/ko/3/tutorial/)
- [W3Schools Python](https://www.w3schools.com/python/)

### Pandas (선택사항)
- [Pandas 10분 입문](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

### 시각화 (선택사항)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/)
- [Plotly Gallery](https://plotly.com/python/)

---

## 💬 자주 묻는 질문

**Q: Python을 배워야 데이터 분석을 잘하나요?**
A: 아니요! 이제는 AI와 협업하는 능력이 더 중요해요. Python은 AI가 대신 써주니까, 여러분은 **문제 정의**와 **해석**에 집중하세요.

**Q: 그래도 코딩을 배우고 싶어요!**
A: 좋아요! AI가 생성한 코드를 보면서 자연스럽게 배울 수 있어요. "왜?"를 계속 물어보세요.

**Q: AI가 틀린 코드를 만들면요?**
A: 에러 메시지를 다시 AI에게 보여주면 고쳐줘요. 여러분은 **결과가 맞는지**만 판단하면 됩니다.

**Q: 나중에 AI 없이도 할 수 있나요?**
A: 이 수업 듣다 보면 자연스럽게 코드가 눈에 익을 거예요. 하지만 AI를 계속 쓰는 게 더 효율적이에요!

---

**기억하세요:**
- **AI 시대의 핵심 역량 = 좋은 질문 + 결과 해석**
- 코딩은 도구일 뿐, 여러분의 도메인 지식이 진짜 무기!
- Python은 "이런게 있구나" 수준이면 충분해요! ✨

---

**다음:** 실전 프롬프트로 진짜 데이터 분석하러 가요! 🚀
