import pandas as pd
from sentence_transformers import SentenceTransformer # SentenceBERT
from sklearn.metrics.pairwise import cosine_similarity # 챗봇의 유사도 계산\
import streamlit as sl
from streamlit_chat import message
from embedding_chatbot.py import embedding
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
#     dc5 = pd.read_csv('chatbot_dataset_t5.csv')
#     dc6 = pd.read_csv('chatbot_dataset_t6.csv')
#     dc7 = pd.read_csv('chatbot_dataset_t7.csv')
#     dc8 = pd.read_csv('chatbot_dataset_t8.csv')
#     dc9 = pd.read_csv('chatbot_dataset_t9.csv')
#     dc10 = pd.read_csv('chatbot_dataset_t10.csv')
    data = pd.concat([data,dc1,dc2,dc3,dc4], ignore_index=True)
    data['embedding'] = data['embedding'].apply(json.loads)
    return data

@sl.cache(allow_output_mutation=True)
def get_answer(model, user_input):
    embedding = model.encode(user_input)

    data['distance'] = data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = data.loc[data['distance'].idxmax()]
    return answer['챗봇']

model = load_model()
data = load_dataset()

sl.header('심리상담 챗봇')

if 'past' not in sl.session_state:
    sl.session_state['past'] = []

if 'generated' not in sl.session_state:
    sl.session_state['generated'] = []
   
placeholder = sl.empty()

with sl.form('form', clear_on_submit=True):
    user_input = sl.text_input('당신: ', '')
    submitted = sl.form_submit_button('전송')

if submitted and user_input:
    answer = get_answer(model, user_input)

    sl.session_state.past.append(user_input)
    sl.session_state.generated.append(answer)

with placeholder.container(): # 리스트에 append된 채팅입력과 로봇출력을 리스트에서 꺼내서 메세지로 출력
        for i in range(len(sl.session_state['past'])):
            message(sl.session_state['past'][i], is_user=True, key=str(i) + '_user')
            if len(sl.session_state['generated']) > i:
                message(sl.session_state['generated'][i], key=str(i) + '_bot')
