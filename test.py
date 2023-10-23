import pandas as pd

data = pd.read_table('steam.txt')

print("데이터의 개수 : ", len(data))

print(data.head())

print("서로 다른 데이터의 수",data['document'].nunique())

# 중복, 결측치 제거
data.drop_duplicates(subset=['document'], inplace=True)
print("데이터의 개수 (중복제거): ", len(data))

print(data.isnull().values.any()) # 결측치 (Null, NaN)가 있다면 True
data = data.dropna(how='any')
print("데이터의 개수 (중복, 결측치 제거): ", len(data))
print()

# document 열에서 한글 문자, 자음, 모음, 공백 문자를 제외한 모든 문자를 삭제하고, 한글 텍스트만 남기기
data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(data[:5])

# label 분류
print(data['label'].value_counts())

# 시각화 작업
import matplotlib.pyplot as plt

# 폰트 맑은 고딕으로 나오게
plt.rcParams['font.family'] = 'Malgun Gothic'

# 0을 '부정', 1을 '긍정'으로 변환
data['label'] = data['label'].replace({0: '부정', 1: '긍정'})

# 변환된 데이터에서 각 값의 개수를 세어 그래프로 표시
data['label'].value_counts().plot(kind='bar', rot=0) # rot은 rotation = 0 으로 하여 글자 가로로 나오게함
plt.xlabel('Document') # x축은 document
plt.ylabel('Label') # y축은 label

# 그래프 시각화
plt.show()