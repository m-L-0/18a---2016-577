#冉文端

#2016011577

导入数据集

```py
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
```

构建距离矩阵函数

```py
def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S
```

构建邻接矩阵函数

```py
def myKNN(x,k):
    n=len(x)
    dis_matrix=np.zeros((n,n))
    w=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            dis_matrix[i][j]=dis_matrix[j][i]=np.sqrt(np.sum((x[i]-x[j])**2))
            #def dist(vec1,vec2):   return np.sqrt(np.sum((vec1-vec2)**2))
    for idx,each in enumerate(dis_matrix):
        index_array=np.argsort(each)
        w[idx][index_array[1:k+1]]=1   
    tmp_w=np.transpose(w)
    w=(tmp_w+w)/2
    return w
```

从拉普拉斯矩阵获得特征矩阵

```py
def getEigVec(L,cluster_num):  
    eigval,eigvec = np.linalg.eig(L)
    dim = len(eigval)
    dictEigval = dict(zip(eigval,range(0,dim)))
    kEig = np.sort(eigval)[0:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix],eigvec[:,ix]
```

标准化的拉普拉斯矩阵

```py
def calLaplacianMatrix(adjacentMatrix):
    degreeMatrix = np.sum(adjacentMatrix, axis=1)
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
```

由距离矩阵找出q个近邻

```py
def q_neighbors(A,q=16):
    n = []
    for i in range(len(A)):
        inds = np.argsort(A[i])
        inds = inds[-q:-1]
        n.append(inds)
    return np.array(n)
```

画出谱聚类图

```py
data = np.array(x)
data = data.T
pos = np.c_[data[2],data[3]]
plt.figure(figsize=(8,8))
G = nx.Graph() 
# 向图G添加节点和边
G.add_nodes_from([i for i in range(150)])
for i in range(len(A)):
    for j in range(len(A)):
        # 只有i是j的近邻且j是i的近邻，二者之间才有边
        if(i in qnn[j] and j in qnn[i]):
            G.add_edge(i,j,weight=A[i,j])
# 画出节点 
nx.draw_networkx_nodes(G, pos, node_color='r', node_size=30, node_shape='o')
# 将图G中的边按照权重分组
edges_list1=[]
edges_list2=[]
edges_list3=[]
for (u,v,d) in G.edges(data='weight'):
    if d > 0.95:
        edges_list1.append((u,v))
    elif d < 0.9:
        edges_list2.append((u,v))
    else:
        edges_list3.append((u,v))
# 按照分好的组，以不同样式画出边
nx.draw_networkx_edges(G, pos, edgelist=edges_list1, width=1, alpha=0.7, edge_color='k', style='solid')
nx.draw_networkx_edges(G, pos, edgelist=edges_list2, width=1, alpha=1.0, edge_color='b', style='dashed')
nx.draw_networkx_edges(G, pos, edgelist=edges_list3, width=1, alpha=0.7, edge_color='g', style='solid')
plt.show()
```

计算正确率

```py
w=myKNN(x,5)
D=getD(w)
L=D-w
clf=KMeans(n_clusters=3,random_state=1)
eigval,eigvec=getEigVec(L,3)
s=clf.fit(eigvec)
C=s.labels_
#计算正确率
count=0
for k in range(150):
    if C[k]==lable[k]:
        count=count+1
acc=float(count)/float(len(y))
print(acc)
```

