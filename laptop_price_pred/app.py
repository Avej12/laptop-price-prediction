import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd
pipe = pickle.load(open('laptop_pipe.pkl','rb'))
df = pd.read_csv('clean_laptop.csv')
st.title('Laptop Price Predictor')

company =st.selectbox('Select Company', df['Company'].unique())
Typename = st.selectbox('Select TypeName', df['TypeName'].unique())
Ram = st.selectbox('Select RAM size in GB',df['Ram'].unique())
Memory = st.selectbox('Select ROM size in GB',[0,128,256,500,512,1000,1024,2048])
gpu = st.selectbox('select GPU',df['Gpu'].unique())
ops = st.selectbox('select Operating System',df['OpSys'].unique())
weight = st.number_input('weight of laptop')
touchscreen = st.selectbox('laptop is touch screen or not',['Yes','No'])
ips =  st.selectbox('laptop is ips or not',['Yes','No'])
screen_size = st.number_input('screen size in Inches')
screen_res = st.selectbox('Screen Resolution is',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
processor = st.selectbox('select processor',df['processor'].unique())
x_res = int(screen_res.split('x')[0])
y_res = int(screen_res.split('x')[1])
ss = st.selectbox('select Storage system',df['SS'].unique())
if st.button('Predict Price'):
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    q = np.array([company,Typename,Ram,Memory,gpu,ops,weight,touchscreen,ips,processor,ppi,ss])
    q = q.reshape(1,12)
    st.title(round(pipe.predict(q)[0],2))