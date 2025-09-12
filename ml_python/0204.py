# 0204.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

#1: load train data
with np.load('./data/0201_data40.npz') as X: # '0201_data50.npz'
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32) #np.float32
    height, width = X['size']

#2: Decision tree : create, train, and predict
#2-1
model = cv2.ml.DTrees_create()
model.setCVFolds(1) # If CVFolds > 1 then, tree pruning using cross-validation is not implemented
model.setMaxDepth(10)
model.setMaxCategories(2) # default:10

#2-2
##model = cv2.ml.Boost_create()
##model.setBoostType(cv2.ml.BOOST_REAL)
##model.setMaxDepth(10)
##model.setWeakCount(100) # default

#2-3
##model = cv2.ml.RTrees_create()
##model.setMaxDepth(100)
##model.setTermCriteria(( cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.01))

#3
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
print("ret=", ret)

#4:
ret, y_pred = model.predict(x_train)
y_pred = y_pred.flatten()
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#5: x_test-> predictions -> pred
step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))

x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
ret, pred = model.predict(x_test)  # pred.shape= (75000, 1)
pred = pred.reshape(xx.shape)      # pred.shape= (250, 300)

#6: display data and result
#6-1
ax = plt.gca()
ax.set_aspect('equal')
#ax.axis('off')
#ax.xaxis.tick_bottom()
#ax.xaxis.tick_top()
#ax.invert_yaxis() # ax.set_ylim(ax.get_ylim()[::-1])

#6-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#6-3
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                20, class_colors[label], 'o')
plt.show()
