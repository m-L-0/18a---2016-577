#姓名  冉文端

#学号  2016011577



导入各种包

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
```

导入鸢尾花数据集

```python
iris = load_iris()
data = iris.data
label = iris.target
```

将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集

```python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,label,test_size=0.2,shuffle=True)
```

创建占位符 ,None代表可选取多个样本数，4表示特征数

```python
xtr = tf.placeholder("float", shape=[None, 4])    
xte = tf.placeholder("float", shape=[4])
```

计算欧式距离

```python
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
```

直接写一个循环选取k值为0到20在循环中设计一个投票表决器对测试集进行预测并计算预测的准确率

```python
with tf.Session() as sess:
    for K in range(21):
        predict=[]
        for i in range(len(Xtest)):
            dist_mat=sess.run(distance,feed_dict={xtr:Xtrain,xte:Xtest[i]})
            knn_idx = np.argsort(dist_mat)[:K]
            classes = [0, 0, 0]
            for idx in knn_idx:
                if(Ytrain[idx]==0):
                    classes[0] += 1
                elif(Ytrain[idx]==1):
                    classes[1] += 1
                else:
                    classes[2] += 1
            Ypred = np.argmax(classes)
            predict.append(Ypred)
        Ypred = predict
        Ytrue = Ytest
        acc = np.sum(np.equal(Ypred,Ytrue)) / len(Ytrue)
        print("K值为"+str(K)+"时准确率为%.2f%%"%(acc*100))
```

可以得到输出结果为

```python
K值为0时准确率为40.00%
K值为1时准确率为100.00%
K值为2时准确率为93.33%
K值为3时准确率为100.00%
K值为4时准确率为100.00%
K值为5时准确率为96.67%
K值为6时准确率为96.67%
K值为7时准确率为96.67%
K值为8时准确率为93.33%
K值为9时准确率为93.33%
K值为10时准确率为93.33%
K值为11时准确率为93.33%
K值为12时准确率为93.33%
K值为13时准确率为93.33%
K值为14时准确率为93.33%
K值为15时准确率为93.33%
K值为16时准确率为93.33%
K值为17时准确率为93.33%
K值为18时准确率为93.33%
K值为19时准确率为93.33%
K值为20时准确率为90.00%
```

可以选取最佳的k值