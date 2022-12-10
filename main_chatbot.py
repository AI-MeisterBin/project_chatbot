import pandas as pd
from sentence_transformers import SentenceTransformer # SentenceBERT
from sklearn.metrics.pairwise import cosine_similarity # 챗봇의 유사도 계산\
import streamlit as sl
from streamlit_chat import message
import json

@sl.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@sl.cache(allow_output_mutation=True)
def load_dataset():
    data = pd.read_csv('chatbot_dataset_v3.csv')
    dc1 = pd.read_csv('chatbot_dataset_t1.csv')
    dc2 = pd.read_csv('chatbot_dataset_t2.csv')
    dc3 = pd.read_csv('chatbot_dataset_t3.csv')
    dc4 = pd.read_csv('chatbot_dataset_t4.csv')
    dc5 = pd.read_csv('chatbot_dataset_t5.csv')
    dc6 = pd.read_csv('chatbot_dataset_t6.csv')
    dc7 = pd.read_csv('chatbot_dataset_t7.csv')
    dc8 = pd.read_csv('chatbot_dataset_t8.csv')
    dc9 = pd.read_csv('chatbot_dataset_t9.csv')
    dc10 = pd.read_csv('chatbot_dataset_t10.csv')
    dc11 = pd.read_csv('chatbot_dataset_t11.csv')
    dc12 = pd.read_csv('chatbot_dataset_t12.csv')
    data = pd.concat([data,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dc8,dc9,dc10,dc11,dc12], ignore_index=True)
    data['embedding'] = data['embedding'].apply(json.loads)
    return data

@sl.cache(allow_output_mutation=True)
def get_answer(model, user_input):
    embedding = model.encode(user_input)

    data['distance'] = data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = data.loc[data['distance'].idxmax()]
    if answer['distance'] > 0.8:
        return answer['챗봇']
    else:
        return "잘 이해하지 못했어요. 좀 더 자세히 말씀 해주시겠어요?"
    
model = load_model()
data = load_dataset()

sl.header('심리상담 챗봇 - 메아리')
sl.subheader('made by 조경빈')

if 'past' not in sl.session_state:
    sl.session_state['past'] = []

if 'generated' not in sl.session_state:
    sl.session_state['generated'] = []
    sl.session_state.generated.append("안녕하세요 당신의 심리상담가 메아리입니다. 편하게 뭐든지 말씀해보세요. ")
   
placeholder = sl.empty()

with sl.form('form', clear_on_submit=True):
    user_input = sl.text_input('당신: ', '')
    submitted = sl.form_submit_button('전송')

if submitted and user_input:
    answer = get_answer(model, user_input)

    sl.session_state.past.append(user_input)
    sl.session_state.generated.append(answer)
    
message(sl.session_state['generated'][0], key=str(0) + '_bot')

with placeholder.container(): # 리스트에 append된 채팅입력과 로봇출력을 리스트에서 꺼내서 메세지로 출력
        for i in range(len(sl.session_state['generated'])-1):
            message(sl.session_state['generated'][i+1], key=str(i+1) + '_bot')
            if len(sl.session_state['past']) > i:
                message(sl.session_state['past'][i], is_user=True, key=str(i) + '_user')
