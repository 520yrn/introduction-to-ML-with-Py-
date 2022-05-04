import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import numpy as np

X,y = mglearn.datasets.make_forge() # 导入分类数据
mglearn.discrete_scatter(X[:,0], X[:,1],y) # 作散点图
plt.legend(['Class 0','Class 1'],loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X shape:{}".format(X.shape))

X,y = mglearn.datasets.make_wave(n_samples = 40) # 导入连续数据.
plt.plot(X,y,'o') # 作散点图。
plt.xlabel("Feature")
plt.ylabel("Target")

cancer = load_breast_cancer() # 分类数据实例:癌症数据
print("cancer.keys:\n{}".format(cancer.keys()))
print("Shape of cancer data:{}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))

boston = load_boston() # 连续数据实例:波士顿房价,原始数据
X,y = mglearn.datasets.load_extended_boston()
print("Data Shape:{}".format(boston.data.shape))
print("X.Shape:{}".format(X.shape))