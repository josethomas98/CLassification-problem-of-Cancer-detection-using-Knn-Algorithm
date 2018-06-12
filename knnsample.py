import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#style.use('fivethirtyeight')
def knn(data,predict,k=3):
    distance=[]
    g={}
    for groups in data:
        g[groups]=0
        for it in data[groups]:
            euclidian=np.sqrt(np.sum(((np.array(it)-np.array(predict))**2)))
            distance.append([euclidian,groups])
    vote=[i[1] for i in sorted(distance)[:k]]
    d=Counter(vote).most_common(1)[0][0]
    return d
dataset={'r':[[1,2],[2,3],[3,1]],'g':[[6,5],[7,7],[8,6]]}
predict=[5,7]
for i in dataset:
    for ii in dataset[i]:
        if(i=='r'):
            v='r'
        else:
            v='g'
        plt.scatter(ii[0],ii[1],color=v)
result=knn(dataset,predict,k=3)

plt.scatter(predict[0],predict[1],s=100,color=result)
plt.show()
        
