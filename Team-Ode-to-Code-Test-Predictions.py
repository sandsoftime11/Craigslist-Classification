#import files 
import csv
test= []
with open("U:/Unstructured Final/Validation/free_for test -Version1.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        test.append(row)

#Drop title
test.pop(0)

#Loading other test result

#DT Model
DT= []
with open("U:/Unstructured Final/Validation/free_DTmodel.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        DT.append(row)

#Drop title
DT.pop(0)

#lr_model
lr= []
with open("U:/Unstructured Final/Validation/free_lr.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        lr.append(row)

#Drop title
lr.pop(0)

#NB
NB= []
with open("U:/Unstructured Final/Validation/free_NBmodel.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        NB.append(row)

#Drop title
NB.pop(0)

#NN
NN= []
with open("U:/Unstructured Final/Validation/free_NNmodel.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        NN.append(row)

#Drop title
NN.pop(0)

#rfc
rfc= []
with open("U:/Unstructured Final/Validation/free_rfc.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        rfc.append(row)

#Drop title
rfc.pop(0)

#SVM
svm= []
with open("U:/Unstructured Final/Validation/free_SVMmodel.csv"
          , "r", encoding='utf-8') as df1:
    csvReader = csv.reader(df1)
    for row in csvReader:
        svm.append(row)

#Drop title
svm.pop(0)

#Create an accuracy test 
score = 0

for i in range(0,len(test)):
    if test[i][2] == DT[i][0]:
        score += 1 
print("DT accuracy score is", round(score/len(test),4))

score = 0
for i in range(0,len(test)):
    if test[i][2] == lr[i][0]:
        score += 1 
print("lr accuracy score is", round(score/len(test),4))

score = 0
for i in range(0,len(test)):
    if test[i][2] == NB[i][0]:
        score += 1 
print("NB accuracy score is", round(score/len(test),4))

score = 0
for i in range(0,len(NN)):
    if test[i][2] == NN[i][0]:
        score += 1 
print("NN accuracy score is", round(score/len(test),4))

score = 0
for i in range(0,len(test)):
    if test[i][2] == rfc[i][0]:
        score += 1 
print("rfc accuracy score is", round(score/len(test),4))

score = 0
for i in range(0,len(test)):
    if test[i][2] == svm[i][0]:
        score += 1 
print("SVM accuracy score is", round(score/len(test),4))

#Try to cutoff

for i in range(0,len(DT)):
    if max(DT[i][1:5]) < '0.25':
        DT[i][0] = '0.0'

score = 0
for i in range(0,len(test)):
    if test[i][2] == DT[i][0]:
        score += 1 
print("DT accuracy score is", round(score/len(test),4))

#####
for i in range(0,len(lr)):
    if max(lr[i][1:5]) < '0.25':
        lr[i][0] = '0.0'

score = 0
for i in range(0,len(test)):
    if test[i][2] == lr[i][0]:
        score += 1 
print("lr accuracy score is", round(score/len(test),4))

#####
for i in range(0,len(NB)):
    if max(NB[i][1:5]) < '0.3':
        lr[i][0] = '0.0'

score = 0
for i in range(0,len(test)):
    if test[i][2] == NB[i][0]:
        score += 1 
print("NB accuracy score is", round(score/len(test),4))


#####
for i in range(0,len(NB)):
    if max(NB[i][1:5]) < '0.3':
        NB[i][0] = '0.0'

score = 0
for i in range(0,len(NN)):
    if test[i][2] == NN[i][0]:
        score += 1 
print("NN accuracy score is", round(score/len(test),4))

#####
for i in range(0,len(rfc)):
    if max(rfc[i][1:5]) < '0.3':
        rfc[i][0] = '0.0'

score = 0
for i in range(0,len(test)):
    if test[i][2] == rfc[i][0]:
        score += 1 
print("rfc accuracy score is", round(score/len(test),4))

#Select the not 'none' item
test_1 = []
index = []
for i in range (0,len(test)):
    if test[i][2] != '5.0':
        test_1.append(test[i])
        index.append(i)
    
#Create an accuracy test for those in the categories 

DT_1=[]
for i in range(0,len(index)):
    DT_1.append(DT[index[i]])

score = 0
for i in range(0,len(test_1)):
    if test_1[i][2] == DT_1[i][0]:
        score += 1 
print("DT accuracy score is", round(score/len(test_1),4))

lr_1=[]
for i in range(0,len(index)):
    lr_1.append(lr[index[i]])

score = 0
for i in range(0,len(test_1)):
    if test_1[i][2] == lr_1[i][0]:
        score += 1 
print("lr accuracy score is", round(score/len(test_1),4))


NB_1=[]
for i in range(0,len(index)):
    NB_1.append(NB[index[i]])

score = 0
for i in range(0,len(test_1)):
    if test_1[i][2] == NB[i][0]:
        score += 1 
print("NB accuracy score is", round(score/len(test_1),4))

index1 = index[:-19]
NN_1=[]
for i in range(0,len(index1)):
    NN_1.append(NN[index1[i]])
score = 0
for i in range(0,len(NN_1)):
    if test_1[i][2] == NN_1[i][0]:
        score += 1 
print("NN accuracy score is", round(score/len(test_1),4))



rfc_1=[]
for i in range(0,len(index)):
    rfc_1.append(rfc[index[i]])


score = 0
for i in range(0,len(test_1)):
    if test_1[i][2] == rfc_1[i][0]:
        score += 1 
print("rfc accuracy score is", round(score/len(test_1),4))



svm_1=[]
for i in range(0,len(index)):
    svm_1.append(svm[index[i]])

score = 0
for i in range(0,len(test_1)):
    if test_1[i][2] == svm_1[i][0]:
        score += 1 
print("SVM accuracy score is", round(score/len(test_1),4))