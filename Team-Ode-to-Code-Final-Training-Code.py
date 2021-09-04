import numpy as np
from sklearn import model_selection, metrics
import xgboost as xgb
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# nltk.download()
from sklearn.model_selection import train_test_split

# df = pd.read_csv(r'C:\Users\sande\Documents\AUD craigslist\CombinedTraining.csv')
assignmentfolder = "C:\\Users\\xinxi\\Purdue\\Spring2021\\MGMT590 UnstructuredData\\Project"

df =  pd.read_csv(assignmentfolder+'\\CombinedTraining.csv')

df = df.drop(columns = ['City', 'Price', 'depth', 'download_timeout', 'download_slot','download_latency'])

df['Text'] = df["Text"].astype(str)
final_col = []
for i in range(len(df['Text'])):
    str_1 = df['Text'][i] + " " + df['Title'][i]
    final_col.append(str_1)


tokenized_text =[]
for row in final_col:
    tokenized_text.append(nltk.word_tokenize(row))

#pre-processing data
lemmatizer = nltk.stem.WordNetLemmatizer()
data_processed=[]
for row in tokenized_text:
  data_processed.append([lemmatizer.lemmatize(token.lower()) for token in row if token not in stopwords.words('english') if token.isalpha() if len(token)>1])


#Tf-Idf
from sklearn.feature_extraction.text import TfidfVectorizer
def identity_tokenizer(text):
  return text
vectorizer1 = TfidfVectorizer(tokenizer=identity_tokenizer, preprocessor= identity_tokenizer, analyzer= 'word', min_df= 2)
vectorizer1.fit(data_processed)
V1 = vectorizer1.transform(data_processed)

#Train-val split and label encoding
x = V1.toarray()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(list(df['Category']))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.3,random_state=123)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


#***************************Input Real Test data*****************************************************
free_data =  pd.read_csv(assignmentfolder+'\\free_for test -Version1.csv')
general_data =  pd.read_csv(assignmentfolder+'\\General_Test.csv')

free_data.columns
free_data = free_data.drop(columns = ['Unnamed: 3', 'Unnamed: 4', 'Furniture', 'Automobile', 'Electronics', 'Cellphone', 'None'])
general_data.columns
general_data = general_data.drop(columns =[ 'City', 'Price', 'depth', 'download_timeout', 'download_slot',
       'download_latency', 'Unnamed: 9', 'Furniture','Cellphones ', 'Automobiles', 'Electronics ', 'None'])

free_data['Text'] = free_data["Text"].astype(str)
final_col_free = []
for i in range(len(free_data['Text'])):
    str_1 = free_data['Text'][i] + " " + free_data['Title'][i]
    final_col_free.append(str_1)


tokenized_text_free =[]
for row in final_col_free:
    tokenized_text_free.append(nltk.word_tokenize(row))

#pre-processing data
lemmatizer = nltk.stem.WordNetLemmatizer()
data_processed_free=[]
for row in tokenized_text_free:
  data_processed_free.append([lemmatizer.lemmatize(token.lower()) for token in row if token not in stopwords.words('english') if token.isalpha() if len(token)>1])

# Here use the trained data's vocabulary:
V1_free = vectorizer1.transform(data_processed_free)

#Train-val split and label encoding
x_free = V1_free.toarray()
print(x_free.shape)
y_free = le.fit_transform(list(free_data['Category']))

print(y_free.shape)


#***************************Start of Models *****************************************************


#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
lr = LogisticRegression(random_state=590)
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_val)
print(classification_report(y_val,y_pred_lr))
print(round(accuracy_score(y_val,y_pred_lr),4))

lr_free_predict = lr.predict(x_free)
lr_free_predict_prob = lr.predict_proba(x_free)

wtr = csv.writer(open (assignmentfolder+'\\free_lr.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(['Prediction' ,'Prob_1', 'Prob_1','Prob_1', 'Prob_1'])
output = np.hstack((np.reshape(lr_free_predict,(-1,1)), lr_free_predict_prob))
for x in output : wtr.writerow (x)


print(classification_report(y_free,lr_free_predict))

#Random forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=590)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_val)
print(classification_report(y_val,y_pred_rfc))
print(round(accuracy_score(y_val,y_pred_rfc),4))

rfc_free_predict = rfc.predict(x_free)
rfc_free_predict_prob = rfc.predict_proba(x_free)
print(classification_report(y_free,rfc_free_predict))


wtr = csv.writer(open (assignmentfolder+'\\free_rfc.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(['Prediction' ,'Prob_1', 'Prob_1','Prob_1', 'Prob_1'])
output = np.hstack((np.reshape(rfc_free_predict,(-1,1)), rfc_free_predict_prob))
for x in output : wtr.writerow (x)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
NBmodel.fit(X_train,y_train)
y_pred_nb = NBmodel.predict(X_val)
print(classification_report(y_val,y_pred_nb))
print(round(accuracy_score(y_val,y_pred_nb),4))

NBmodel_free_predict = NBmodel.predict(x_free)
print(classification_report(y_free,NBmodel_free_predict))

NBmodel_free_predict_prob = NBmodel.predict_proba(x_free)

wtr = csv.writer(open (assignmentfolder+'\\free_NBmodel.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(['Prediction' ,'Prob_1', 'Prob_1','Prob_1', 'Prob_1'])
output = np.hstack((np.reshape(NBmodel_free_predict,(-1,1)), NBmodel_free_predict_prob))
for x in output : wtr.writerow (x)

#Support Vector Machine
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
SVMmodel.fit(X_train,y_train)
y_pred_svm = SVMmodel.predict(X_val)
print(classification_report(y_val,y_pred_svm))
print(round(accuracy_score(y_val,y_pred_svm),4))

SVMmodel_free_predict = SVMmodel.predict(x_free)
print(classification_report(y_free,SVMmodel_free_predict))

SVMmodel_free_predict_prob = SVMmodel.decision_function(x_free)

wtr = csv.writer(open (assignmentfolder+'\\free_SVMmodel.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(['Prediction' ,'Distance_1', 'Distance_2','Distance_3', 'Distance_4'])
output = np.hstack((np.reshape(SVMmodel_free_predict,(-1,1)), SVMmodel_free_predict_prob))
for x in output : wtr.writerow (x)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTmodel = DecisionTreeClassifier()
DTmodel.fit(X_train,y_train)
y_pred_dt = DTmodel.predict(X_val)
print(classification_report(y_val,y_pred_dt))
print(round(accuracy_score(y_val,y_pred_dt),4))

DTmodel_free_predict = DTmodel.predict(x_free)
print(classification_report(y_free,DTmodel_free_predict))

DTmodel_free_predict_prob = DTmodel.predict_proba(x_free)

wtr = csv.writer(open (assignmentfolder+'\\free_DTmodel.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(['Prediction' ,'Prob_1', 'Prob_1','Prob_1', 'Prob_1'])
output = np.hstack((np.reshape(DTmodel_free_predict,(-1,1)), DTmodel_free_predict_prob))
for x in output : wtr.writerow (x)

#Neural Network
from sklearn.neural_network import MLPClassifier
NNmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,2), random_state=1)
NNmodel.fit(X_train,y_train)
y_pred_nn = NNmodel.predict(X_val)
print(classification_report(y_val,y_pred_nn))
print(round(accuracy_score(y_val,y_pred_nn),4))

NNmodel_free_predict = NNmodel.predict(x_free)
print(classification_report(y_free,NNmodel_free_predict))

NNmodel_free_predict_prob = NNmodel.predict_proba(x_free)

wtr = csv.writer(open (assignmentfolder+'\\free_NNmodel.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(['Prediction' ,'Prob_1', 'Prob_1','Prob_1', 'Prob_1'])
output = np.hstack((np.reshape(NNmodel_free_predict,(-1,1)), NNmodel_free_predict_prob))
for x in output : wtr.writerow (x)