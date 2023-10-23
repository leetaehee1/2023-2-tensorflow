import pandas as pd

data = {
    'label': [],
    'document': []
}

with open('steam.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().split('\t')  # 가정: 탭으로 열을 구분
        if len(line) == 2:
            data['label'].append(int(line[0]))
            data['document'].append(line[1])

df = pd.DataFrame(data)
df.to_csv('steam.txt', sep='\t', index=False)
# print(df)