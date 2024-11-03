import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

data_path='C:\\Users\\Bartosz Kebel\\Desktop\\Machine Learning\\Lista 2\\mnist.npz'
data_set = np.load( data_path, allow_pickle=True)
train_labels=data_set['train_labels']
test_lables=data_set['test_labels']
train_data=data_set['train_data']
test_data=data_set['test_data']


def preproces (data_set):
    data_set = np.array([record.flatten() for record in data_set])
    return data_set/255


def random_10_proc_sample(data_set,data_labels):
    indexes=np.arange(len(data_set))
    indexes=np.random.choice(indexes,len(data_set)*10//100,replace=False)
    return (data_set[indexes],data_labels[indexes])


result=[[] for i in range(10)]

'''
test_data=preproces(test_data)

for k in range(1,11):
    
    bootstrap_iterations = [1, 2, 11, 22, 44, 88 , 100 ]
    for boot_iter in bootstrap_iterations:
        bagging = BaggingClassifier(
        estimator=KNeighborsClassifier(k, weights='distance'),
        n_estimators=boot_iter, 
        bootstrap=True)
    

        train_sample, train_sample_lables=random_10_proc_sample(train_data,train_labels)
        train_sample = preproces(train_sample)

        bagging.fit(train_sample,train_sample_lables)
        predictions=bagging.predict(test_data)
        result[k-1].append((boot_iter,accuracy_score(test_lables,predictions)))
        print(f'done k == {k} for {boot_iter} iterations')
print(result)

np.save('result.txt' , np.array(result))
'''

'''
result=np.load('result.txt.npy')
for i in range(10):

    x_points=[result[i][j][1] for j in range(len(result[i]))]
    y_points=[result[i][j][0] for j in range(len(result[i]))]
    plt.plot(y_points,x_points, marker='o' , label=f'k={i+1}')

plt.legend()
plt.show()
'''

# zad 2

'''
result=[[] for i in range(10)]
for k in range(10):
    knn=KNeighborsClassifier(k+1, weights='distance')
    train_sample, train_sample_lables=random_10_proc_sample(train_data,train_labels)
    train_sample = preproces(train_sample)
    counter=0
    for i in range(len(train_sample)):
        mask=np.ones(train_sample_lables.shape,bool)
        mask[i]=False
        knn.fit(train_sample[mask],train_sample_lables[mask])
        
        predict=knn.predict([train_sample[i]])
        right_one=train_sample_lables[i]
        predict_num=predict[0]
        if ( predict_num == right_one):
            counter+=1


    result[k]=counter/len(train_sample)
    print(f'done for k = {k}')

print(result)
np.save('result2',np.array(result))
'''

result=np.load('result2.npy')
kys=[i+1 for i in range(10)]
plt.bar(kys,result)
plt.ylim(0.9, 1)
plt.show()


