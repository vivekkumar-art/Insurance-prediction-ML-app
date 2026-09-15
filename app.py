import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st
#this is for web based application project



#Web page Code 
st.tittle("HEALTH INSURENCE PREDICTION")
img_url="https://imgs.search.brave.com/TFjx_njIOx9kqjKDS6hW0MHtdM52dYbdvYuqHBihhwk/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly90aHVt/YnMuZHJlYW1zdGlt/ZS5jb20vYi9oZWFs/dGgtaW5zdXJhbmNl/LWJ1c2luZXNzbWFu/LWRyYXdpbmctbGFu/ZGluZy1wYWdlLXdo/aXRlLWJhY2tncm91/bmQtNzE3ODQ0NTYu/anBn"
st.image(img_url)

#Load data and ML Model Part
url="https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"
df=pd.read_csv(url)
df.sample()

#Step 3: EDA: Exporatory data analysis
df.head()
df.drop("Customer_ID",axis=1,inplace=True)

df["Previous_Insurance"]=df["Previous_Insurance"].map({"Yes":1,"No":0})
df["Insurance_Bought"]=df["Insurance_Bought"].map({"Yes":1,"No":0})

#step 4: Divide into features and traget
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#Step 5: Divide data into Training and testinf part
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.3)  #test_size=0.3 means 30 % data


#Step 6 :Train model
model=LogisticRegression()
model.fit(X_train,y_train)

#show data sample
st.write(df.head())
#Create bar for user input form
st.sidebar.tittle("Fill Customer Details")
st.sidebar.image(img_url)

#to get user input
all_ans=[]
for index,col_name in enumerate(X.columns):
  min_v = X[col_name].min()
  max_v=X[col_name].max()
  if col_name!="Previous_Insurance";
    value= st.slidebar.slider(f"Select value for {col_name}",
                              min_value=min_v,
                              max_value=max_v)
else:
  value = st.slidebar.number_input(f"Select values for {col_name}:")

all_ans.append(value)

ud ={j:all_ans[i] for i.j in enumerate(X.columns)}
user_df = pd.DataFrame(ud,index=[1])
st.write(user_df)

#=========================================Precdtion================================================================
if st.button("Click to predict:"):
  with st.spinner("Predicting..")
  import time
  time.sleep(2)
final_ans=model.predict([all_ans])[0]
if final_ans==0:
  st.info("❎Customer will not buy the insurance❎")
else:
  st.success("✅Customer will buy the insurance✅")



