import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random
def knn(data,predict,k=3):
    distance=[]
    for groups in data:
        for it in data[groups]:
            euclidian=np.sqrt(np.sum(((np.array(it)-np.array(predict))**2)))
            distance.append([euclidian,groups])
    vote=[i[1] for i in sorted(distance)[:k]]
    d=Counter(vote).most_common(1)[0][0]
    confidence=Counter(vote).most_common(1)[0][1]/k
    return d,confidence
df=pd.read_csv('C:/Users/Stino Thomas/Desktop/machine learning dataset/breast cancer.txt')
df.replace('?',-9999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)
test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])
correct=0
total=len(test_data)
c=0
name={2:" benign",4:" malignant"}
for group in test_set:
    for ii in test_set[group]:
        vote,conf=knn(train_set,ii,3)
        #print("the vote as for ",name[vote],"  the actual is ",name[group],"confidence: ",conf)
        if(vote==group):
            correct+=1
        c=c+1
r=[float(i) for i in range(0,9)]
s=["Clump Thickness: 1 - 10 ","Uniformity of Cell Size","Uniformity of Cell Shape: 1 - 10 "," Marginal Adhesion: 1 - 10 "," Single Epithelial Cell Size: 1 - 10 ","Bare Nuclei: 1 - 10 "," Bland Chromatin: 1 - 10 "
," Normal Nucle: 1 - 10 "," Mitoses: 1 - 10 "]
for i in range(0,len(s)):
    r[i]=float(input(s[i]))
vote,confi=knn(train_set,r,5)
print("The result is ",name[vote]," prediction is : ",confi*100)
#print("total : ",c,"correct : ",correct)
print("accuracy: ",correct*100/total,"%")
