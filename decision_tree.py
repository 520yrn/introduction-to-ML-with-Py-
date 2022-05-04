from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import mglearn

cancer = cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify = cancer.target, random_state = 42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train) # 训练未剪枝决策树
print("train set score: {:.3f}".format(tree.score(X_train,y_train)))
print("Test set score: {:.3f}".format(tree.score(X_test,y_test)))

tree = DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(X_train,y_train) # 训练预剪枝决策树,约定最大深度为4
print("train set score: {:.3f}".format(tree.score(X_train,y_train)))
print("Test set score: {:.3f}".format(tree.score(X_test,y_test)))

export_graphviz(tree,out_file = "tree.dot",class_names=['malingant','benign'],
                feature_names=cancer.feature_names,impurity=False,filled=True) # 作图
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).render('决策树可视化') # 决策树可视化

print("Feature importances:\n{}".format(tree.feature_importances_)) # 观察决策树特征重要性(介于0~1,越大越好)
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importances")
    plt.ylabel("Feature")
plot_feature_importances_cancer(tree) # 特征重要性水平柱状图可视化
tree = mglearn.plots.plot_tree_not_monotone()