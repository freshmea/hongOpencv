# 0203.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

#1: load train data
with np.load('data/0201_data40.npz') as X:   # '0201_data50.npz'
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32)    #np.float32
    height, width = X['size']

#2: k-nearest neighbours: create, train, and predict
#2-1
model = cv2.ml.KNearest_create()
tm = cv2.TickMeter()
tm.start()
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
tm.stop()
print("train time: %fms" % (tm.getTimeMilli()))
tm.reset()
#2-2: x_test-> predictions -> pred
step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))

x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
k = 3 # 1, 3, 5
tm.start()
ret, pred = model.predict(x_test, k) # pred.shape= (75000, 1)
tm.stop()
print("predict time: %fms" % (tm.getTimeMilli()/75000))
tm.reset()
pred = pred.reshape(xx.shape)        # pred.shape= (250, 300)

#3: display data and result
#3-1
ax = plt.gca()
ax.set_aspect('equal')
#ax.axis('off')
#ax.xaxis.tick_bottom()
#ax.xaxis.tick_top()
#ax.invert_yaxis() # ax.set_ylim(ax.get_ylim()[::-1])

#3-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#3-3
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                20, class_colors[label], 'o')
plt.show()
