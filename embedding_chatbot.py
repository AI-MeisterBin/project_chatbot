import pandas as pd
from sentence_transformers import SentenceTransformer # SentenceBERT
from sklearn.metrics.pairwise import cosine_similarity # 챗봇의 유사도 계산

model = SentenceTransformer('jhgan/ko-sroberta-multitask') # transformer모듈

def embedding(model):
  # 심리상담 데이터 불러오기
  data_psy = pd.read_csv('wellness_dataset.csv',encoding = 'CP949')

  # 전처리
  #data_psy = data_psy.drop(columns=['Unnamed: 3']) # 쓸모없는 데이터 제거
  data_psy = data_psy[~data_psy['챗봇'].isna()] # 오류가 날만한 NaN데이터 제거

  # 대화 데이터 불러오기
  data_chat = pd.read_csv('ChatbotData.csv', encoding = 'CP949')

  #data로 병합
  data = pd.concat([data_psy,data_chat], ignore_index=True)
  
  # 임베딩을 저장할 더미데이터
  data['embedding'] = pd.Series([[]] * len(data))
  
  # 임베딩에 유저열에 있는 데이터를 임베딩해서 넣기
  data['embedding'] = data['유저'].map(lambda x: list(model.encode(x)))

  return data

data = embedding(model)
data.to_csv('chatbot_dataset_v13.csv', index=False) # 전처리한 데이터 저장
