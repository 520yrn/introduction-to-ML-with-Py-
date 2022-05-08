import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC

mglearn.plots.plot_scaling() # 展示不同的数据预处理方法
cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify = cancer.target, random_state = 1)
print(X_train.shape)
print(X_test.shape)

Scaler = MinMaxScaler() # 标准化处理:x-new=(x-x_min)/(x_max-x_min)
Scaler.fit(X_train) # 只通过X训练数据确定预处理用的统计量
X_train_scaled = Scaler.transform(X_train) # 变换数据
print("Transformed shape:{}".format(X_train_scaled.shape))
print("Pre-feature minimum before scaling:\n{}".format(X_train.min(axis = 0)))
print("Pre-feature maximum before scaling:\n{}".format(X_train.max(axis = 0)))
print("Pre-feature minimum after scaling:\n{}".format(X_train_scaled.min(axis = 0)))
print("Pre-feature maximum after scaling:\n{}".format(X_train_scaled.max(axis = 0)))
X_test_scaled = Scaler.transform(X_test) # 变换测试数据
print("Pre-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis = 0)))
print("Pre-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis = 0)))

X,_ = make_blobs(n_samples=50,centers=5,random_state=4,cluster_std=2)
X_train,X_test = train_test_split(X,random_state = 5,test_size = .1)
fig, axes = plt.subplots(1,3,figsize=(13,4))
axes[0].scatter(X_train[:,0],X_train[:,1],c = mglearn.cm2(0),label = "Training set",s = 60)
axes[0].scatter(X_test[:,0],X_test[:,1],c = mglearn.cm2(1),label = "Test set",s = 60,marker = "^")
axes[0].legend(loc = 'upper left')
axes[0].set_title("original data")
Scaler = MinMaxScaler() # 标准化处理:x-new=(x-x_min)/(x_max-x_min)
Scaler.fit(X_train) # 只通过X训练数据确定预处理用的统计量
X_train_scaled = Scaler.transform(X_train) # 变换数据
X_test_scaled = Scaler.transform(X_test) # 变换数据
axes[1].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c = mglearn.cm2(0),label = "Training set",s = 60)
axes[1].scatter(X_test_scaled[:,0],X_test_scaled[:,1],c = mglearn.cm2(1),label = "Test set",s = 60,marker = "^")
axes[1].legend(loc = 'upper left')
axes[1].set_title("scaled data")

Scaler = MinMaxScaler() # 标准化处理:x-new=(x-x_min)/(x_max-x_min)
Scaler.fit(X_test) # 只通过X训练数据确定预处理用的统计量
X_test_scaled_badly = Scaler.transform(X_test) # 变换数据(单独变换test数据集是错误的!)
axes[2].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c = mglearn.cm2(0),label = "Training set",s = 60)
axes[2].scatter(X_test_scaled_badly[:,0],X_test_scaled_badly[:,1],c = mglearn.cm2(1),label = "Test set",s = 60,marker = "^")
axes[2].legend(loc = 'upper left')
axes[2].set_title("Improperly scaled data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

''' from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X)
    X_scaled = scaler.fit_transform(X) # mean-std标准化,代码更加简洁'''

cancer = load_breast_cancer() # 分类数据实例:癌症数据    
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, random_state = 0)
svm = SVC(C=100)
svm.fit(X_train,y_train)
print("Test set score: {:.3f}".format(svm.score(X_test,y_test)))
Scaler = MinMaxScaler() # 标准化处理:x-new=(x-x_min)/(x_max-x_min)
Scaler.fit(X_train) # 只通过X训练数据确定预处理用的统计量
X_train_scaled = Scaler.transform(X_train) # 变换数据
X_test_scaled = Scaler.transform(X_test) # 变换数据
svm.fit(X_train_scaled,y_train)
print("Test set score: {:.3f}".format(svm.score(X_test_scaled,y_test))) # 标准化后精度提高
