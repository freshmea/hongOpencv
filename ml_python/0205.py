# 0205.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)

#1: load train data
with np.load('data/0201_data40.npz') as X: # '0201_data50.npz'
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32) #np.float32
    height, width = X['size']

#2: NormalBayesClassifier : create, train, and predict
#2-1
model =  cv2.ml.NormalBayesClassifier_create()
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)

#2-2:
ret, y_pred = model.predict(x_train)
y_pred = y_pred.flatten()
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#2-3: x_test-> predictions -> pred
step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))
x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
ret, pred = model.predict(x_test)

#2-4:
##ret, pred, prob = model.predictProb(x_test)
##prob_sum = np.sum(prob, axis=1)
##prob[:, 0] /= prob_sum
##prob[:, 1] /= prob_sum
##print("prob=",prob)

#2-5:
pred = pred.reshape(xx.shape) #pred.shape= (250, 300)

#3: display data and result
#3-1
ax = plt.gca()
ax.set_aspect('equal')

#3-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#3-3
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                20, class_colors[label], 'o')
plt.show()
