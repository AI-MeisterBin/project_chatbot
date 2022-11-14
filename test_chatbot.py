import pandas as pd
from sentence_transformers import SentenceTransformer # SentenceBERT
from sklearn.metrics.pairwise import cosine_similarity # 챗봇의 유사도 계산

model = SentenceTransformer('jhgan/ko-sroberta-multitask') # transformer모듈

data = pd.read_pickle('chatbot_dataset_v2.pkl') # 미리 저장한 데이터 불러오기

text = '나 너무 힘들어'

embedding = model.encode(text)

data['similarity'] = data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) # 유사도 text와 유사도 측정

answer = data.loc[data['similarity'].idxmax()] # 유사도가 제일 높은 데이터를 선정

# 항목별로 출력
print('구분 :', answer['구분'])
print('유사한 질문 :', answer['유저'])
print('챗봇 답변 :', answer['챗봇'])
print('유사도 :', answer['similarity'])
