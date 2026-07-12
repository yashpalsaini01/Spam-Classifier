import streamlit as st
import pandas as pd
import string

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(vectorizer_path, "rb") as f:
    tfidf = pickle.load(f)




print(model)
print(hasattr(model, "classes_"))

st.title("Email/SMS Spam Classifier")
input=st.text_input("Enter your email/sms")

def trans(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return y

if st.button("Prediction"):

    if input.strip() == "":
        st.warning("Please enter an Email/SMS.")
    else:
        tran = " ".join(trans(input))
        vector = tfidf.transform([tran])
        res = model.predict(vector)[0]

        if res == 1:
            st.error("Spam Detected")
        else:
            st.success("Not Spam")