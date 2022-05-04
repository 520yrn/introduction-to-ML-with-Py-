'''
例1:鸢尾花分类。
输入:花萼长度/宽度,花瓣长度/宽度(单位:cm)
输出:花的类别(setosa/versicolor/virginica)
'''
import pandas as pd
import numpy as np
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))
print("Target names:{}".format(iris_dataset['target_names']))
print("Feature names:{}".format(iris_dataset['feature_names']))
print("Type of data:{}".format(type(iris_dataset['data'])))
print("shape of data:{}".format(iris_dataset['data'].shape))
print("Type of target:{}".format(type(iris_dataset['target'])))
print("shape of target:{}".format(iris_dataset['target'].shape))

X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state = 0)
# 划分数据集,75%用于训练,25%用于测试。
print("X-train shape :{}".format(X_train.shape))
print("X-test shape :{}".format(X_test.shape))
print("y-train shape :{}".format(y_train.shape))
print("y-test shape :{}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train,columns = iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe,c = y_train,figsize=(15,15),marker="o",
                        hist_kwds={'bins':20},s=60,alpha=.8,cmap = mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train) # k邻近模型训练
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new) # 预测
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:{}".format(y_pred))
print("Test set score:{}".format(knn.score(X_test,y_test))) # 计算预测精度