from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import mglearn

X,y = mglearn.datasets.make_forge() # 导入分类数据

fig, axes = plt.subplots(1,2,figsize=(10,3))
for model, ax in zip([LinearSVC(), LogisticRegression()],axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps = 0.5, ax = ax, alpha = 0.7) # 作决策边界
    mglearn.discrete_scatter(X[:,0], X[:,1], y ,ax = ax) # 标数据散点
    ax.set_title(" {} ".format(clf.__class__.__name__))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend()
mglearn.plots.plot_linear_svc_regularization()

cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify = cancer.target, random_state = 42)
logreg = LogisticRegression().fit(X_train,y_train)
print("train set score: {:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test,y_test)))
logreg100 = LogisticRegression(C=100).fit(X_train,y_train) # C越大,正则化越强,学习训练集能力越强(过拟合)
print("train set score: {:.3f}".format(logreg100.score(X_train,y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test,y_test)))
logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
print("train set score: {:.3f}".format(logreg001.score(X_train,y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test,y_test)))

plt.figure(3)
plt.plot(logreg.coef_.T,'s',label = "C=1") # .T表示转置
plt.plot(logreg100.coef_.T,'^',label = "C=100")
plt.plot(logreg001.coef_.T,'v',label = "C=0.01")
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation = 90)
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel("coef index")
plt.ylabel("coef magnitude")
plt.legend()

plt.figure(4)
# 以上默认使用L2正则化方法,下面演示L1正则化方法
for C,marker in zip([0.001,1,100],['o','^','v']):
    lr_L1 = LogisticRegression(C=C,solver='liblinear',penalty="l1").fit(X_train,y_train)
    print("Train accuracy of l1 logreg with C={:.3f} : {:.2f}".format(C,lr_L1.score(X_train,y_train)))
    print("Test accuracy of l1 logreg with C={:.3f} : {:.2f}".format(C,lr_L1.score(X_test,y_test)))
    plt.plot(lr_L1.coef_.T,marker,label = "C={:.3f}".format(C)) # .T表示转置
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation = 90)
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel("coef index")
plt.ylabel("coef magnitude")
plt.legend(loc = 3)

plt.figure(5)
X,y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(["Class 0","Class 1","Class 2"])
linear_svm = LinearSVC().fit(X,y) # 多分类学习,实际对每类样本点产生分类式
print("Coefficient shape",linear_svm.coef_.shape)
print("Intercept shape",linear_svm.intercept_.shape)

line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color) # y=0,求两特征表达式
plt.xlim(-10,8)
plt.ylim(-10,15)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(['Class 0','Class 1','Class 2','Line class 0','Line class 1','Line class 2'],loc = (1.01,0.3))
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha = 0.7) # 作决策边界