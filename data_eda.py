import pandas as pd

data = pd.read_table('ratings_train2.txt')

print("데이터의 개수 : ", len(data))

print(data.head())

print("서로 다른 데이터의 수",data['document'].nunique())

data.drop_duplicates(subset=['document'], inplace=True)
print("데이터의 개수 (중복제거): ", len(data))

import matplotlib.pyplot as plt

print(data['label'].value_counts())
data['label'].value_counts().plot(kind='bar')
# plt.show()

# 결측치 제거
print(data.isnull().values.any()) # 결측치 (Null, NaN)가 있다면 True
data = data.dropna(how='any')
print("데이터의 개수 (결측치 제거): ", len(data))
print()

data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(data[:5])