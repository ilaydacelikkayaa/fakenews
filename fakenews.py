import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#Bu sınıf, metin verilerini sayısal verilere dönüştürmek için kullanılır
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#Multinomial Naive Bayes, çoklu sınıflandırma problemleri için uygundur ve özellikle kelime frekansı gibi özellikler kullanıldığında etkili bir modeldir.
data=pd.read_csv("fake_or_real_news.csv")
data
x=np.array(data['title'])
y=np.array(data['label'])
cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)
import streamlit as st
st.title("Fake News Detection System")
def fakenewsdetection():
    user=st.text_area("Enter Any News Headline:")
    if len(user)<1:
        st.write("Please enter a news")
    else:
        sample=user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
fakenewsdetection()
