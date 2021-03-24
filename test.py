# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# import matplotlib.pyplot as plt
# iris = load_iris()
# X = iris.data[:,[2, 3]]
# y = iris.target
# clf = LogisticRegression()
# clf.fit(X, y)
# x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
# y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),
#                      np.arange(y_min,y_max, 0.1))
# xx_ravel = xx.ravel()
# yy_ravel = yy.ravel()
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.plot()
# plt.contourf(xx, yy, Z, alpha=0.4, cmap = plt.cm.RdYlBu)
# plt.scatter(X[:, 0], X[:, 1], c=y,  cmap = plt.cm.brg)
# plt.title("Logistic Regression")
# plt.xlabel("Petal.Length")
# plt.ylabel("Petal.Width")
# plt.show()
#

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


# 绘制分界线
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def LR_test():
    # 读入数据
    X, Y = load_planar_dataset()

    reg = sklearn.linear_model.LogisticRegression()
    reg.fit(X.T, Y.T)

    plot_decision_boundary(lambda x: reg.predict(x), X, Y)  # 绘制决策边界
    plt.title("Logistic Regression")  # 图标题
    plt.show()

    LR_predictions = reg.predict(X.T)  # 预测结果
    print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
                                   np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          "% " + "(正确标记的数据点所占的百分比)")

LR_test()
print(1)
