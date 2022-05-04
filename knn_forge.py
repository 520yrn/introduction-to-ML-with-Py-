import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
import numpy as np

X,y = mglearn.datasets.make_forge() # 导入分类数据
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("Test set prediction:{}".format(clf.predict(X_test)))
print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))

fig, axes = plt.subplots(1,3,figsize=(10,3))
for n_neighbors, ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps = 0.5, ax = ax, alpha = 0.4) # 作决策边界
    mglearn.discrete_scatter(X[:,0], X[:,1], y ,ax = ax) # 标数据散点
    ax.set_title("{}neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc = 3)

cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify = cancer.target, random_state = 66)
train_accuracy = []
test_accuracy = []
neighbors_setting = range(1,11) # neighbors取值1到10
for n_neighbors in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train,y_train)
    train_accuracy.append(clf.score(X_train,y_train)) # 记录训练精度
    test_accuracy.append(clf.score(X_test,y_test)) # 记录预测精度
plt.figure(2)
plt.plot(neighbors_setting,train_accuracy,label = "train_accuracy")
plt.plot(neighbors_setting,test_accuracy,label = "test_accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.legend()

X,y = mglearn.datasets.make_wave(n_samples = 40) # 导入连续数据.
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
print("Test set predictions:\n{}".format(reg.predict(X_test)))
print("Test set R^2:{:.2f}".format(reg.score(X_test,y_test)))

fig, axes = plt.subplots(1,3,figsize=(15,4))
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9],axes): # 利用1,3,9个邻居分别预测
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize = 8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize = 8)
    ax.set_title("{} neighbor(s)\n train score{:.2f} test score{:.2f}".format(
        n_neighbors,reg.score(X_train,y_train),reg.score(X_test,y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions","Training data/target","Test data/target"])