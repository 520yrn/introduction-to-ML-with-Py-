import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles

display(mglearn.plots.plot_logistic_regression_graph())
display(mglearn.plots.plot_single_hidden_layer_graph())
display(mglearn.plots.plot_two_hidden_layer_graph())

line = np.linspace(-3,3,100)
plt.plot(line,np.tanh(line),label = "tanh")
plt.plot(line,np.maximum(line,0),label = "tanh")
plt.legend(loc = "best")
plt.xlabel("x")
plt.ylabel("relu(x),tanh(x)")
plt.show() # 作变幻函数图

X,y = make_moons(n_samples = 100,noise = 0.25,random_state = 3)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y, random_state = 42)
mlp = MLPClassifier(solver = 'lbfgs',random_state=0,hidden_layer_sizes=[10,5],activation = "tanh")
# 运用多层感知机,hidden_layer_size控制NN大小,隐藏层1设为10(默认100),隐藏层2设为5
# 运用多层感知机,activation控制激活函数
mlp.fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill = True, alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

fig, axes = plt.subplots(2,4,figsize = (20,8))
for axx,n_hidden_nodes in zip(axes,[10,100]):
    for ax,alpha in zip(axx,[0.0001,0.01,0.1,1]): # alpha控制正则化
            mlp = MLPClassifier(solver = 'lbfgs',random_state=0,
                                hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],alpha=alpha)
            mlp.fit(X_train,y_train)
            mglearn.plots.plot_2d_separator(mlp, X_train,fill=True, ax=ax,alpha=0.3)
            mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
            ax.set_title("n_hidden=[{},{}]\n alpha={:.4f}".format(n_hidden_nodes,n_hidden_nodes,alpha))

fig, axes = plt.subplots(2,4,figsize = (20,8))
for i,ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver = 'lbfgs',random_state=i,hidden_layer_sizes=[100,100])
    # 因为权重是随机选取的,所以随机种子影响学习结果
    mlp.fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train,fill=True, ax=ax,alpha=0.3)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
    
cancer = load_breast_cancer() # 分类数据实例:癌症数据
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, random_state = 0)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
print("train set score: {:.3f}".format(mlp.score(X_train,y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test,y_test)))

mean_on_train = X_train.mean(axis = 0)
std_on_train = X_train.std(axis = 0) # 获取特征均值与标准差
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(max_iter = 1000,random_state=0) # 增加最大迭代次数max_iter
mlp.fit(X_train_scaled,y_train)
print("train set score: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test_scaled,y_test)))

plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation = None,cmap = 'viridis') # 作学习特征热力图
plt.yticks(range(30),cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

X,y = make_circles(noise=0.25,factor = 0.5,random_state=1)
y_named = np.array(["blue","red"])[y] # 重命名标签名
X_train,X_test,y_train_named,y_test_named,y_train,y_test = train_test_split(X,y_named, y, random_state = 0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train) # 构建梯度提升树
print("X_test.shape: {}".format(X_test.shape))
print("decision function.shape: {}".format(gbrt.decision_function(X_test).shape))
print("decision function:\n {}".format(gbrt.decision_function(X_test)[:6])) 
# 查看前几个预测置信度,decision function大于0为正类,小于0为反类

fig, axes = plt.subplots(1,2,figsize = (13,5))
mglearn.tools.plot_2d_separator(gbrt, X,ax = axes[0],alpha = .4,fill = True,cm = mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbrt, X,ax = axes[1],alpha = .4,cm = mglearn.ReBl)
# 作分类器置信度图
for ax in axes:
    mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test,markers = '^',ax=ax)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,markers = 'o',ax=ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
cbar = plt.colorbar(score_image,ax = axes.tolist())
axes[0].legend(["test class 0","test class 1","train class 0","train class 1"],ncol = 4,loc = (.1,1.1))

print("shape of probability:{}".format(gbrt.predict_proba(X_test).shape))
print("Predicted probabilities:{}".format(gbrt.predict_proba(X_test)[:6])) 
# 分类器预测概率,前者为第一类,后者为第二类,两类概率和为1,超过50%的为预测结果