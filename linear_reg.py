import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np

X,y = mglearn.datasets.make_wave(n_samples = 60)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42)
lr = LinearRegression().fit(X_train,y_train)
print("lr.coef_:{}".format(lr.coef_)) # y = aX+b中向量a
print("lr.intercept_:{}".format(lr.intercept_)) # 截距
print("Train set score {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score {:.2f}".format(lr.score(X_test,y_test)))

X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
lr = LinearRegression().fit(X_train,y_train)
print("Train set score {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score {:.2f}".format(lr.score(X_test,y_test)))

ridge = Ridge().fit(X_train,y_train) # 岭回归,L2正则化在优化函数后加上参数平方和
print("Train set score {:.2f}".format(ridge.score(X_train,y_train)))
print("Test set score {:.2f}".format(ridge.score(X_test,y_test)))
ridge10 = Ridge(alpha = 10).fit(X_train,y_train)
print("Train set score {:.2f}".format(ridge10.score(X_train,y_train)))
print("Test set score {:.2f}".format(ridge10.score(X_test,y_test)))
ridge01 = Ridge(alpha = 0.1).fit(X_train,y_train)
print("Train set score {:.2f}".format(ridge01.score(X_train,y_train)))
print("Test set score {:.2f}".format(ridge01.score(X_test,y_test)))

plt.plot(ridge.coef_,'s',label = "Ridge alpha=1")
plt.plot(ridge10.coef_,'^',label = "Ridge alpha=10")
plt.plot(ridge01.coef_,'v',label = "Ridge alpha=0.1")
# 作系数图,x[i]对应第i个特征的系数,以此类推。

plt.plot(lr.coef_,'o',label = "LinearRegression")
plt.xlabel("coef index")
plt.ylabel("coef magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()

plt.figure(2)
mglearn.plots.plot_ridge_n_samples()
# 作学习曲线,横坐标表示训练集的大小,纵坐标为模型优劣的衡量。

lasso = Lasso().fit(X_train, y_train) # lasso回归,L1正则化在优化函数后加上参数绝对值
print("Train set score {:.2f}".format(lasso.score(X_train,y_train)))
print("Test set score {:.2f}".format(lasso.score(X_test,y_test)))
print("Number of feature used: {}".format(np.sum(lasso.coef_ != 0)))
lasso001 = Lasso(alpha = 0.01, max_iter=100000).fit(X_train, y_train) # 控制超参以及迭代次数
print("Train set score {:.2f}".format(lasso001.score(X_train,y_train)))
print("Test set score {:.2f}".format(lasso001.score(X_test,y_test)))
print("Number of feature used: {}".format(np.sum(lasso001.coef_ != 0)))
lasso00001 = Lasso(alpha = 0.0001, max_iter=100000).fit(X_train, y_train) # 控制超参以及迭代次数
print("Train set score {:.2f}".format(lasso00001.score(X_train,y_train)))
print("Test set score {:.2f}".format(lasso00001.score(X_test,y_test)))
print("Number of feature used: {}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(lasso.coef_,'s',label = "lasso alpha=1")
plt.plot(lasso001.coef_,'^',label = "lasso alpha=10")
plt.plot(lasso00001.coef_,'v',label = "lasso alpha=0.1")
plt.plot(ridge01.coef_,'o',label = "Ridge alpha=0.1")
# 作系数图,x[i]对应第i个特征的系数,以此类推。

plt.figure(3)
plt.ylim(-25,25)
plt.xlabel("coef index")
plt.ylabel("coef magnitude")
plt.legend(ncol = 2,loc = (0,1.05))