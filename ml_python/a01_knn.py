# 0202.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

#1: load train data
with np.load('./data/0201_data50.npz') as X:
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32) #np.float32
    height, width = X['size']
##x_train = np.float32(x_train)
##y_train = np.int32(y_train)
##print("x_train=", x_train)
##print("y_train=", y_train)
##print("height=", height)
##print("width=", width)

#2: k-nearest neighbours: create, train, and predict
#2-1
model = cv2.ml.KNearest_create()
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)

#2-2
x_test = np.array([[100, 200], [200, 250], [300, 300]], dtype = np.float32)
k = 3   # 1, 3, 5
ret, y_pred = model.predict(x_test, k)
print("y_pred=", y_pred)

#2-3
ret, y_pred2, neighbours, dists = model.findNearest(x_test, k)
print("y_pred2=", y_pred2)
print("neighbours=", neighbours)
print("dists=", dists)

#3: display data and result
#3-1
class_colors = ['blue', 'red']
y_pred = np.int32(y_pred.flatten())

#3-2
ax = plt.gca()
ax.set_aspect('equal')
ax.xaxis.tick_bottom()
#ax.axis('off')
#ax.xaxis.tick_top()
#ax.invert_yaxis() # ax.set_ylim(ax.get_ylim()[::-1]) # the same as OpenCV

plt.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], 20, class_colors[0], 'o')
plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], 20, class_colors[1], 'o')

plt.scatter(x_test[y_pred==0, 0], x_test[y_pred==0, 1], 50, c=class_colors[0], marker='x')
plt.scatter(x_test[y_pred==1, 0], x_test[y_pred==1, 1], 50, c=class_colors[1], marker='x')

#3-3
##for label in range(2): # 2 class
##    plt.scatter(*x_train[y_train==label, :].T, s = 20, marker= 'o', c = class_colors[label])
##    plt.scatter(*x_test[y_pred==label, :].T, s = 50, marker= 'x', c = class_colors[label])

#3-4
##for label in range(2): # 2 class
##    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1], 20, class_colors[label], 'o')
##    plt.scatter(x_test[y_pred==label, 0], x_test[y_pred==label, 1], 50, c=class_colors[label], marker='x')

plt.xlim(0, width-1)
plt.ylim(0, height-1)
plt.show()
