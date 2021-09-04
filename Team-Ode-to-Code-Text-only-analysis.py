import nltk
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from sklearn import model_selection, metrics
import xgboost as xgb
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# nltk.download()
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\sande\Documents\AUD craigslist\CombinedTraining.csv')

df = df.drop(columns = ['Title', 'City', 'Price', 'depth', 'download_timeout', 'download_slot','download_latency'])

df['Text'] = df["Text"].astype(str)
Text = df['Text'].apply(nltk.word_tokenize).tolist()

#pre-processing data
lemmatizer = nltk.stem.WordNetLemmatizer()
text_processed=[]
for row in Text:
  text_processed.append([lemmatizer.lemmatize(token.lower()) for token in row if token not in stopwords.words('english') if token.isalpha() if len(token)>1])

#Tf-Idf
from sklearn.feature_extraction.text import TfidfVectorizer
def identity_tokenizer(text):
  return text
vectorizer1 = TfidfVectorizer(tokenizer=identity_tokenizer, preprocessor= identity_tokenizer, analyzer= 'word', min_df= 2)
vectorizer1.fit(text_processed)
Text_V1 = vectorizer1.transform(text_processed)

#Train-val split and label encoding
x = Text_V1.toarray()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(list(df['Category']))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=123)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
lr = LogisticRegression(random_state=590)
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_val)
print(classification_report(y_val,y_pred_lr))
print(round(accuracy_score(y_val,y_pred_lr),4))

#Random forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=590)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_val)
print(classification_report(y_val,y_pred_rfc))
print(round(accuracy_score(y_val,y_pred_rfc),4))

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
NBmodel.fit(X_train,y_train)
y_pred_nb = NBmodel.predict(X_val)
print(classification_report(y_val,y_pred_nb))
print(round(accuracy_score(y_val,y_pred_nb),4))

#Support Vector Machine
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
SVMmodel.fit(X_train,y_train)
y_pred_svm = SVMmodel.predict(X_val)
print(classification_report(y_val,y_pred_svm))
print(round(accuracy_score(y_val,y_pred_svm),4))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTmodel = DecisionTreeClassifier()
DTmodel.fit(X_train,y_train)
y_pred_dt = DTmodel.predict(X_val)
print(classification_report(y_val,y_pred_dt))
print(round(accuracy_score(y_val,y_pred_dt),4))

#Neural Network
from sklearn.neural_network import MLPClassifier
NNmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,2), random_state=1)
NNmodel.fit(X_train,y_train)
y_pred_nn = NNmodel.predict(X_val)
print(classification_report(y_val,y_pred_nn))
print(round(accuracy_score(y_val,y_pred_nn),4))