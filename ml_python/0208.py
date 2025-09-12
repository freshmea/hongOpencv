# 0208.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)

#1: load train data
with np.load('data/0201_data50.npz') as X:
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.float32)
    height, width = X['size']

#2: K-means clustering
K = 3 # 3
term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, labels, centers = cv2.kmeans(data=x_train, K= K, bestLabels=None,
                                  criteria=term_crit,
                                  attempts= 1,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)
print('centers=', centers)
print('centers.shape=', centers.shape)
print('labels.shape=', labels.shape)
print('ret=', ret)

#3:
def predict(pts, centers):
    dist = np.sqrt(((pts - centers[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(dist, axis=0)

step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))
x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
pred = predict(x_test, centers) # pred.shape = (75000,)

#4: display data using matplotlib
#4-1
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(0, width-1)
plt.ylim(0, height-1)

#4-2
pred = pred.reshape(xx.shape)  # pred.shape = (250, 300)
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#4-3
class_colors = ['blue', 'green', 'cyan']
labels = labels.flatten()
for i in range(K):
    pts = x_train[labels==i]
    ax.scatter(pts[:, 0], pts[:, 1], 20, c=class_colors[i], marker='o')
    ax.scatter(centers[:, 0], centers[:, 1], 100, c='red', marker='x')
plt.show()
