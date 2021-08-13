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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#import numpy as np
#from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
tfidf_v=TfidfVectorizer()
#deploy = deploy.replace(r'^\s*$', '', regex=True)
deploy["Conditions"]=deploy["Condition"].astype('str')
deploy.iloc[:,0].astype('str')
X=deploy["Reviews"]
y=deploy["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=365)
model= Pipeline([('tfidf', TfidfVectorizer()),
                         ('log', LogisticRegression(penalty='l2',multi_class='auto',class_weight = 'balanced',solver = 'liblinear',intercept_scaling=.005)),])
     
        
model.fit(X_train,y_train)
y_testpred=model.predict(X_test)
y_trainpred=model.predict(X_train)
   
prediction = model.predict(y)
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
              (df["GENDER"].values[:,None]==deploy['Sex'].values),deploy["Prediction"])
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



