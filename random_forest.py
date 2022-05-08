from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importances")
    plt.ylabel("Feature")
    
X,y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train,X_test,y_train,y_test = train_test_split(X, y, stratify = y, random_state = 42)
forest = RandomForestClassifier(n_estimators=5,random_state=2)
# 参数n_estimators控制树的个数
forest.fit(X_train,y_train)

fig, axes = plt.subplots(2,3,figsize = (20,10))
for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train,fill=True, ax=axes[-1,-1],alpha=0.4)
axes[-1,-1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:,0], X_train[:,1],y_train)

cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, random_state = 0)
forest = RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(X_train,y_train)
print("train set score: {:.3f}".format(forest.score(X_train,y_train)))
print("Test set score: {:.3f}".format(forest.score(X_test,y_test)))
plt.figure(2)
plot_feature_importances_cancer(forest) # 查看特征重要性

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, random_state = 0)
gbrt = GradientBoostingClassifier(random_state = 0) # 梯度提升树
gbrt.fit(X_train,y_train)
print("train set score: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Test set score: {:.3f}".format(gbrt.score(X_test,y_test)))

gbrt = GradientBoostingClassifier(random_state = 0,learning_rate=0.01) # 控制学习率预剪枝
gbrt.fit(X_train,y_train)
print("train set score: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Test set score: {:.3f}".format(gbrt.score(X_test,y_test)))

gbrt = GradientBoostingClassifier(random_state = 0,max_depth=1) # 控制最大深度预剪枝
gbrt.fit(X_train,y_train)
plt.figure(3) # 查看特征重要性
plot_feature_importances_cancer(gbrt)
print("train set score: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Test set score: {:.3f}".format(gbrt.score(X_test,y_test)))
