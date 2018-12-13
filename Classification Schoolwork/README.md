#冉文端

#2016011577

导入此任务所需要的各种

```python
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from spectral import *
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
```

从文件夹中导入数据集

```python
data = np.array([])
target = np.array([])
n=0
for i in os.listdir('9个类别的数据集-train/'):
    a,b=os.path.splitext(i)
    new=sio.loadmat('9个类别的数据集-train/'+i)[a]
    for j in range(new.shape[0]):
        target=np.append(target,n)
    data = np.append(data,new)
    n=n+1
data = np.reshape(data,(-1,200))
target = np.reshape(target,(-1,1))
```

划分数据集为训练集和测试集8:2

```python
xtrain,xdev,ytrain,ydev=train_test_split(data,target,test_size=0.2)
```

数据预处理 数据归一化

```PYTHON
scaler=StandardScaler()
scaler.fit(xtrain)
xtrain=scaler.transform(xtrain)
xdev=scaler.transform(xdev)
```

用SVM训练模型 进行格点搜索 选取一个最优参数

```py
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {"gamma":[1, 0.1, 0.01],"C":[0.1, 1, 10]}
grid_search = GridSearchCV(SVC(),param_grid,cv=5) 
grid_search.fit(xtrain,ytrain.ravel()) 
```

利用上步选取的最优参数 构建学习模型并对模型的性能进行预测

```py
from sklearn.metrics import accuracy_score
svc = SVC(C=10,gamma=0.01,class_weight='balanced')
svc.fit(xtrain,ytrain)
ypred = svc.predict(xdev)
print(accuracy_score(ydev,ypred))
#0.9523465703971119
```

导入测试集 并进行数据预处理

```py
test_data = sio.loadmat("data_test_final.mat")['data_test_final']
test_data = np.array(test_data, dtype=np.float64)
xtest = scaler.transform(test_data)
```

对测试集进行预测并将预测标签导出为csv文件进行保存

```pytho
import pandas as pd
ytest = svc.predict(xtest)
data = pd.DataFrame(ytest)
data.to_csv("ytest.csv")
```

