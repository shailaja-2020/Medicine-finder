# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
import numpy as np
deploy = pd.read_excel('Deployproject_Data.xlsx')
deploy = deploy.dropna()

st.title('Medicine Finder')

st.sidebar.header('User Input Parameters')
 
def user_input_features():
   
    CONDITION = st.sidebar.selectbox('CONDITION',deploy["Condition"].astype('str'))
    DRUG= st.sidebar.selectbox('DRUG',deploy["Drug"].astype('str'))
    AGE = st.sidebar.selectbox('AGE',deploy['Age'].astype('str'))
    SEX= st.sidebar.selectbox('SEX',deploy["Sex"].astype('str'))
   
    
    data = {'CONDITION':CONDITION,
            'DRUG':DRUG,
            'AGE':AGE,
           'GENDER':SEX,}
    features = pd.DataFrame(data,index = [0])
    return features 

    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
a=df["CONDITION"]
b=df["DRUG"]
c=df["AGE"]
d=df["GENDER"]

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from imblearn.over_sampling import SMOTE


tfidf_v=TfidfVectorizer()
X = tfidf_v.fit_transform(deploy['Reviews'].values.astype('U')) 
y=deploy.Sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=365)
sm=SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model_smooth=LogisticRegression( )
model_smooth.fit(X_train_res,y_train_res)        
y_testpred=model_smooth.predict(X_test)
y_trainpred=model_smooth.predict(X_train)
  
prediction = model_smooth.predict(X)
deploy["Overall_sentiment"]=prediction


rev=np.dot(
              (df["CONDITION"].values[:,None]==deploy['Condition'].values)&
              (df["DRUG"].values[:,None]==deploy['Drug'].values)&
              (df["AGE"].values[:,None]==deploy['Age'].values)&
              (df["GENDER"].values[:,None]==deploy['Sex'].values),deploy["Reviews"])
sent=np.dot(
              (df["CONDITION"].values[:,None]==deploy['Condition'].values)&
              (df["DRUG"].values[:,None]==deploy['Drug'].values)&
              (df["AGE"].values[:,None]==deploy['Age'].values)&
              (df["GENDER"].values[:,None]==deploy['Sex'].values),deploy["Sentiment"])

predict=np.dot(
              (df["CONDITION"].values[:,None]==deploy['Condition'].values)&
              (df["DRUG"].values[:,None]==deploy['Drug'].values)&
              (df["AGE"].values[:,None]==deploy['Age'].values)&
              (df["GENDER"].values[:,None]==deploy['Sex'].values),deploy["Overall_sentiment"])
rating=np.dot(
              (df["CONDITION"].values[:,None]==deploy['Condition'].values)&
              (df["DRUG"].values[:,None]==deploy['Drug'].values)&
              (df["AGE"].values[:,None]==deploy['Age'].values)&
              (df["GENDER"].values[:,None]==deploy['Sex'].values),(deploy["EaseofUse"]+deploy["Effectiveness"]+deploy["Satisfaction"])/3)
              
              
st.subheader('Review for the drug')
st.write(rev)


st.subheader('Vader classification')
st.write(sent)

st.subheader('Prediction')
st.write(predict)


st.subheader('Rating')
st.write(rating)



