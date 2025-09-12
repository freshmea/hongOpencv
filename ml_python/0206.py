# 0206.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

#np.set_printoptions(precision=2, suppress=True)

#1: load train data
with np.load('data/0201_data40.npz') as X: # '0201_data50.npz'
    X_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.float32)
    height, width = X['size']

#2: normalize
def calulateStat(X):
    mu = np.mean(X, axis=0)
    var= np.var(X,  axis=0)
    return mu, var

def normalizeScale(X, mu, var):

    eps = 0.00001
    X_hat = (X-mu)/(np.sqrt(var + eps))
    return X_hat

mu, var = calulateStat(X_train)
print("mu=", mu)
print("var=", var)
x_train = normalizeScale(X_train, mu, var)

##from sklearn.preprocessing import StandardScaler
##scaler = StandardScaler()
##x_train = scaler.fit_transform(X_train)

#3: a logistic regression classifier
#3-1
model = cv2.ml.LogisticRegression_create()
#model.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH) # _BATCH
#model.setMiniBatchSize(4)
#model.setIterations(100)
#model.setLearningRate(0.001) # default alpha = 0.001
#model.setRegularization(cv2.ml.LogisticRegression_REG_L2)
#crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.001)
#model.setTermCriteria(crit)

ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses= y_train)

#3-2
ret, y_pred = model.predict(x_train)
y_pred = y_pred.flatten()
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#3-3:
thetas = model.get_learnt_thetas()[0] # theta[0] + x*theta[1] + y*theta[2]
print("thetas=", thetas)

x_train_t = cv2.hconcat([np.ones((x_train.shape[0], 1), dtype=x_train.dtype),
                         x_train]) # [1, x, y]

x = np.dot(x_train_t, thetas)

# from scipy.special import expit
def sigmoid(x):
    return  1./(1. + np.exp(-x))
y = sigmoid(x)
y_pred2 = np.int32(y>0.5) # y_pred
accuracy2 = np.sum(y_train == y_pred2) / len(y_train)
print('accuracy2=', accuracy2) # accuracy

#4: x_test-> predictions -> pred
step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))

X_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
#x_test = scaler.transform(X_test) # sklearn
x_test = normalizeScale(X_test, mu, var)

ret, pred = model.predict(x_test)
pred = pred.reshape(xx.shape)

#5: display data and result
#5-1
ax = plt.gca()
ax.set_aspect('equal')

#5-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#5-3
for label in range(2): #2 is number of class
    plt.scatter(X_train[y_train==label, 0], X_train[y_train==label, 1],
                20, class_colors[label], 'o')

plt.show()
