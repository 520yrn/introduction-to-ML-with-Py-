from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import axes3d,Axes3D
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import mglearn
import numpy as np

X,y = make_blobs(centers =4,random_state = 8)
y = y % 2
mglearn.discrete_scatter(X[:,0],X[:,1],y) # 作散点图
plt.xlabel("feature 0")
plt.ylabel("feature 1")

plt.figure(2)
linear_svm = LinearSVC().fit(X,y) # 学习数据
mglearn.plots.plot_2d_classification(linear_svm, X) # 观察分类结果
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# 添加第二个特征的平方作为一个新特征
X_new = np.hstack([X,X[:,1:]**2])
figure = plt.figure()
# 3D可视化,首先画出所有y=0的点,再画出所有y=1的点
ax = Axes3D(figure,elev=-152,azim=-26)
mask = y == 0
ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker = '^',cmap=mglearn.cm2,s=60)
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature 1 ** 2")

linear_svm_3d = LinearSVC().fit(X_new,y) # 拟合扩展后数据
coef,intercept = linear_svm_3d.coef_.ravel(),linear_svm_3d.intercept_

figure = plt.figure()
ax = Axes3D(figure,elev=-152,azim=-26)
xx = np.linspace(X_new[:,0].min()-2,X_new[:,0].max()+2,50)
yy = np.linspace(X_new[:,1].min()-2,X_new[:,1].max()+2,50)
XX,YY = np.meshgrid(xx,yy)
ZZ = (coef[0]*XX+coef[1]*YY+intercept) / -coef[2] # 计算平面上的点
ax.plot_surface(XX,YY,ZZ,rstride=8,cstride = 8,alpha=0.3) # 作拟合平面图

ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker = '^',cmap=mglearn.cm2,s=60)
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature 1 ** 2")

plt.figure(6)
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(),YY.ravel(),ZZ.ravel()])
plt.contour(XX,YY,dec.reshape(XX.shape),levels = [dec.min(),0,dec.max()],cmap = mglearn.cm2,alpha=0.5)
# 观察平面上的分割图
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

plt.figure(7)
X,y =mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel = 'rbf',C=10,gamma=0.1).fit(X,y) 
# rbf表示高斯核函数,直接计算新空间上点之间的距离(内积),C正则化参数,限制重要性
# gamma控制点与点之间的邻近距离(定义邻近)
mglearn.plots.plot_2d_separator(svm,X,eps=.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
sv = svm.support_vectors_ # 画支持向量
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth = 3)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

fig,axes = plt.subplots(3,3,figsize=(15,10))
for ax,C in zip(axes,[-1,0,3]):
    for a,gamma in zip(ax,range(-1,2)):
        mglearn.plots.plot_svm(log_C=C,log_gamma=gamma,ax=a) # 调参,观察参数影响
axes[0,0].legend(["class 0","class 1","sv class 0","sv class 1"],ncol=4,loc=(.9,1.2))

cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, random_state = 0)
svc = SVC()
svc.fit(X_train,y_train) # SVC拟合
print("train set score: {:.3f}".format(svc.score(X_train,y_train)))
print("Test set score: {:.3f}".format(svc.score(X_test,y_test)))

plt.figure(9)
plt.plot(X_train.min(axis = 0),'o',label='min')
plt.plot(X_train.max(axis = 0),'^',label='max')
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log") # 测试各指标量级

min_on_training = X_train.min(axis = 0)
range_on_training = (X_train - min_on_training).max(axis = 0)
X_train_scaled = (X_train - min_on_training)/range_on_training
X_test_scaled = (X_test - min_on_training)/range_on_training
print("min for each feature \n{}".format(X_train_scaled.min(axis = 0)))
print("max for each feature \n{}".format(X_train_scaled.max(axis = 0)))
# max-min标准化,可以去除量级影响
svc = SVC()
svc.fit(X_train_scaled,y_train) # SVC拟合
print("train set score: {:.3f}".format(svc.score(X_train_scaled,y_train)))
print("Test set score: {:.3f}".format(svc.score(X_test_scaled,y_test)))




