
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn. feature_extraction. text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import nltk
from nltk.util import pr
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
stopword=set (stopwords.words ('english'))
stemmer = nltk.SnowballStemmer ("english")

data=pd.read_csv('twitter_data.csv')
print(data.head())

data.shape

data.info()

plt.pie(data['class'].value_counts().values,
        labels = data['class'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

data["labels"] = data["class"].map({0: "Hate Speech", 
                                    1: "Offensive Language", 
                                    2: "No Hate and Offensive"})
print(data.head())



data = data[["tweet", "labels"]]
print(data.head())

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

sample = "Im a boss ass bitch"" ...that needs to sit in a corner"
data = cv.transform([sample]).toarray()
print(clf.predict(data))

#!pip install streamlit

def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user= st.text_area("Enter any Tweet:")
    if len(user)<1:
      st.write(" ")
    else:
         sample=user
         data=cv.transform([sample]).toarray()
         a=clf.predict(data)
         st.title(a)
hate_speech_detection()


