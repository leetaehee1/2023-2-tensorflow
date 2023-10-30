# <p align="center"> KOELECTRA를 활용한 스팀 게임 리뷰 감성 분석 </p>
<p align="center"><img src="https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/95b2d3fe-9bc0-4584-8b5f-47b3995bfe27" width="750px" height="380px"></p>  

# 1. 개요
이번 프로젝트에서는 한국어 자연어 처리 모델인 [KOELECTRA](https://github.com/monologg/KoELECTRA)를 활용하여 **[스팀](https://namu.wiki/w/Steam)** 게임 리뷰의 긍부정 예측을 하고자 한다.  

## 1.1 문제 정의

**게임 리뷰**는 비디오 게임에 대한 평가, 분석, 및 의견을 담은 글, 비디오, 오디오, 또는 다른 형태의 콘텐츠이다. 
이러한 게임 리뷰는 게임 소비자에게 게임의 내용과 질에 대한 정보를 제공하며, 이를 통해 소비자는 자신의 관심사와 기대치에 부합하는 게임을 선택할 수 있다. 또한, 다른 게임과 비교하여 그들의 취향과 목적을 가장 잘 충족시킬지 비교하여 선택하는데 도움을 준다.

이런 게임 리뷰를 통해 사람들은 리뷰를 읽고 게임의 내용과 품질을 평가한 후 게임을 구매할지 여부를 결정하거나, 재미요소, 게임플레이, 스토리, 그래픽 품질 등을 파악하고 잘 작성된 리뷰를 통해 잘못 선택한 게임으로 인한 시간과 자원의 낭비를 피하고 **자신이 즐길만한 만족스러운 게임**을 선택할 수 있다.

이로 인해 게임 리뷰는 게임 커뮤니티와 산업에 중요한 역할을 하며, 게임 소비자와 게임 개발자 간의 상호작용은 게임의 품질 향상과 그들의 작품을 개선하고 발전시키는데 도움을 준다.




<!-- - 게임 리뷰와 게임 흥행의 상관관계를 나타내는 참고자료를 통해 게임 리뷰의 가치를 언급
 - [[1]](링크) 등의 표기를 문장 뒤에 활용하여 참고 문헌을 작성할 것
 - 프로젝트로 해당 과업을 해결할 때 기대할 수 있는 장점, 활용 가능성 등을 언급
 - 필요에 따라서는 적절한 그림을 그려 표현(ppt 등) -->

## 1.2 데이터 및 모델 개요
데이터는 **bab2min**의 Github - corpus 에서 공유하는 [스팀 게임 리뷰](https://github.com/bab2min/corpus/tree/master/sentiment)를 활용[2]하여, 총 10만 건의 데이터에 대해서 사전 학습 언어 모델의 재학습(fine-tuning)을 수행한다.   
게임 유통 서비스인 Steam의 각종 게임에 달린 한국어 리뷰를 수집한 것이다. 게임 커뮤니티 특성 상 비속어 및 은어가 많이 사용된 것이 특징이다. 데이터는 탭으로 분리되어 있으며, 첫번째 필드에는 긍/부정(1=긍정, 0=부정), 두번째 필드에는 리뷰 텍스트가 위치합니다. 긍정과 부정의 비율이 1:1에 가깝도록 샘플링된 자료이다.

 - 언어: 한국어
 - 출처: Steam (https://store.steampowered.com/)
 - 수집 기간: 2020.05~2020.06
### 데이터 분포

| |건수|
|---|---|
|긍정|49,996|
|부정|50,004|
|합계|100,000|

## 1.3 데이터 미리보기

| 입력          |모델|출력|
|-------------|---|---|
|**Document**|---|**label**|
| 스팀 게임 리뷰 문장|[KOELECTRA small](https://github.com/monologg/KoELECTRA) [3]|부정(0), 긍정(1) |
|노래가 너무 적음|---|0|
|돌겠네 진짜. 황숙아, 어크 공장 그만 돌려라. 죽는다.|---|0|
|막노동 체험판 막노동 하는사람인데 장비를 내가 사야돼 뭐지|---|1|
|차악!차악!!차악!!! 정말 이래서 왕국을 되찾을 수 있는거야??|---|1|
|시간 때우기에 좋음.. 도전과제는 50시간이면 다 깰 수 있어요|---|1|
|역시 재미있네요 전작에서 할수 없었던 자유로운 덱 빌딩도 좋네요^^|---|1|
|재미있었습니다.|---|1|  

# 2. 데이터
## 감성 분석 순서  
![634](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/4837a454-600b-4542-a845-4f29be85674b)



## 2.1 전체 데이터 소스
```
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
print("데이터의 개수 (결측치 제거): ", len(data))
print()

# document 열에서 한글 문자, 자음, 모음, 공백 문자를 제외한 모든 문자를 삭제하고, 한글 텍스트만 남기기
data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(data[:5])

# label 분류
print(data['label'].value_counts())

# 시각화 작업
import matplotlib.pyplot as plt

# 변환된 데이터에서 각 값의 개수를 세어 그래프로 표시
data['label'].value_counts().plot(kind='bar', rot=0) # rot은 rotation = 0 으로 하여 글자 가로로 나오게함
plt.xlabel('Document') # x축은 document
plt.ylabel('Label') # y축은 label

# 그래프 시각화
plt.show()
```

## 2.2 탐색적 데이터 분석

## 2.3 데이터 전처리
- 입력 데이터의 전처리 과정
  
  텍스트 파일을 읽어 데이터프레임으로 변환 -> 탭으로 열 구분 -> 데이터프레임을 텍스트파일로 저장  
  *document와 label의 열을 두개가 아닌 하나로 인식하기 때문*
  ```
  import pandas as pd

  data = {
      'label': [],
      'document': []
  }
  
  with open('steam.txt', 'r', encoding='utf-8') as file:
      for line in file:
          line = line.strip().split('\t')  # 탭으로 열을 구분
          if len(line) == 2:
              data['label'].append(int(line[0]))
              data['document'].append(line[1])
  
  df = pd.DataFrame(data)
  df.to_csv('steam.txt', sep='\t', index=False)
  # print(df)
  ```
  
- 전체 데이터의 양
  ```
  print("데이터의 개수 : ", len(data))
  ```
  ![data_size](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/38f37b78-12bd-42f5-aeb4-09d8bd444eaa)
  
- 데이터 중복, 결측치 제거
  ```
  data.drop_duplicates(subset=['document'], inplace=True)
  print("데이터의 개수 (중복제거): ", len(data))
  
  print(data.isnull().values.any()) # 결측치 (Null, NaN)가 있다면 True
  data = data.dropna(how='any')
  print("데이터의 개수 (결측치 제거): ", len(data))
  print()
  ```
  ![del_nullandDupl](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/1300b005-3fa6-41ca-ad31-ca720d1583b9)

- 특수문자나 숫자 제거후 **한글 텍스트만** 남기기
  ```
  # document 열에서 한글 문자, 자음, 모음, 공백 문자를 제외한 모든 문자를 삭제하고, 한글 텍스트만 남기기
  data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")  
  ```
  
- **label** 분류 및 시각화
  ```
  import matplotlib.pyplot as plt
  
  print(data['label'].value_counts())
  data['label'].value_counts().plot(kind='bar')
  plt.show()
  ```
  ![labelcases](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/91d96113-805f-4f19-9659-76e2a2c2d14c)  
  ![Figure_1](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/471992fb-6f27-4f90-af89-82b791d9b396)

- 총 10만개의 데이터 중 학습할 데이터 1만개 추출
  ```
  import random

  import pandas as pd
  
  data = pd.read_table('steam.txt')
  print(data)
  
  # 데이터프레임의 행 수를 가져옵니다
  num_rows = len(data)
  
  # 1만개의 무작위 행 인덱스를 생성
  random_indices = random.sample(range(num_rows), 10000)
  
  # 무작위로 선택된 행을 추출
  random_sample = data.iloc[random_indices]
  
  # 인덱스를 오름차순으로 정렬
  random_sample = random_sample.sort_index(ascending=True)
  print(random_sample)

  # 데이터 저장
  random_sample.to_csv('steam_10000.txt', index=False)
  ```

- 데이터 확인  
  ![data_10000](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/62daf4f8-21df-4dee-aa2c-77d86a95ac02)


- 학습과 검증 데이터셋 분리    
  ![데이터셋분리](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/dc9625fb-78ea-4812-b2d8-fdf27f063df7)

- 학습 데이터의 구성  
  학습 데이터 10개
  ![image](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/9e12696c-f853-4bfc-8dcf-e3aa829c2850)

  검증 데이터 10개  
  ![labeldata](https://github.com/leetaehee1/Koelectra_SteamReview/assets/79897716/337faf0b-3f82-4121-91ac-d48e1c43e9f8)

# 3. 재학습 결과

## 3.1 개발 환경
 - pycharm, ...
 - [requirements.txt](https://github.com/leetaehee1/Koelectra_SteamReview/blob/main/requirements.txt)
## 3.2 KOELECTRA fine-tuning
## 3.3 학습 결과 그래프

# 4. 배운점


## 참고문헌
steam 이미지 : https://cdn.cloudflare.steamstatic.com/store/home/store_home_share.jpg  
[2] Github / bab2min - corpus의 Steam.txt 자료 : https://github.com/bab2min/corpus/tree/master/sentiment  
[3] Koelectra 모델에 대한 참고문헌 : https://github.com/monologg/KoELECTRA
