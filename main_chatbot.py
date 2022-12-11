import pandas as pd
from sentence_transformers import SentenceTransformer # SentenceBERT
from sklearn.metrics.pairwise import cosine_similarity # 챗봇의 유사도 계산\
import streamlit as sl
from streamlit_chat import message
import json

@sl.cache(allow_output_mutation=True)
def load_model(): # 모델 로드
    model = SentenceTransformer('AI-MeisterBin/ko-sentence-bert-MeisterBin')
    return model

@sl.cache(allow_output_mutation=True)
def load_dataset(): # 데이터 로드
    data = pd.read_csv('chatbot_dataset_well.csv')
    dc1 = pd.read_csv('chatbot_dataset_f1.csv')
    dc2 = pd.read_csv('chatbot_dataset_f2.csv')
    dc3 = pd.read_csv('chatbot_dataset_f3.csv')
    dc4 = pd.read_csv('chatbot_dataset_f4.csv')
    dc5 = pd.read_csv('chatbot_dataset_f5.csv')
    dc6 = pd.read_csv('chatbot_dataset_f6.csv')
    dc7 = pd.read_csv('chatbot_dataset_f7.csv')
    dc8 = pd.read_csv('chatbot_dataset_f8.csv')
    dc9 = pd.read_csv('chatbot_dataset_f9.csv')
    dc10 = pd.read_csv('chatbot_dataset_f10.csv')
    dc11 = pd.read_csv('chatbot_dataset_f11.csv')
    dc12 = pd.read_csv('chatbot_dataset_f12.csv')
    data = pd.concat([data,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dc8,dc9,dc10,dc11,dc12], ignore_index=True) # 데이터 합치기
    data['embedding'] = data['embedding'].apply(json.loads)
    return data

@sl.cache(allow_output_mutation=True)
def get_answer(model, user_input): # 입력에 따라 답 출력
    embedding = model.encode(user_input) # 입력 임베딩

    data['distance'] = data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) # 학습 데이터와 비교
    answer = data.loc[data['distance'].idxmax()] # 제일 유사도가 높은 값 저장
    if answer['distance'] > 0.7: # 유사도가 0.8 이상이면 답 출력
        return answer['챗봇']
    else: 
        return "잘 이해하지 못했어요. 좀 더 자세히 말씀 해주시겠어요?"

# 함수 불러오기
model = load_model()
data = load_dataset()

# 사이트의 헤더와 보조헤더 추력
sl.header('심리상담 챗봇 - 메아리') #챗
sl.subheader('made by 조경빈')

# 유저의 입력과 챗봇의 답변을 저장
if 'past' not in sl.session_state:
    sl.session_state['past'] = []

if 'generated' not in sl.session_state:
    sl.session_state['generated'] = []
    #첫 문구 작성
    sl.session_state.generated.append("안녕하세요 당신의 심리상담가 메아리입니다. 편하게 뭐든지 말씀해보세요. ")

# 입력 칸을 아래로 내리기
placeholder = sl.empty()

# 입력 칸 디자인
with sl.form('form', clear_on_submit=True):
    user_input = sl.text_input('당신: ', '')
    submitted = sl.form_submit_button('전송')

# 답변 횟수를 통해 답변 출력 후 유저의 입력과 챗봇의 답변을 저장
if submitted and user_input:
    answer = get_answer(model, user_input)

    sl.session_state.past.append(user_input)
    sl.session_state.generated.append(answer)
    
#message(sl.session_state['generated'][0], key=str(0) + '_bot')

with placeholder.container(): # 리스트에 저장된 유저입력과 로봇출력을 리스트에서 꺼내서 메세지로 출력
        for i in range(len(sl.session_state['generated'])):
            message(sl.session_state['generated'][i], key=str(i) + '_bot')
            if len(sl.session_state['past']) > i:
                message(sl.session_state['past'][i], is_user=True, key=str(i) + '_user')
