# 0207.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

#1: load train data
with np.load('data/0201_data40.npz') as X: # '0201_data50.npz'
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32) #np.float32
    height, width = X['size']

#2: Support Vector Machines
#2-1
model =  cv2.ml.SVM_create()
#model.setType(cv2.ml.SVM_C_SVC) # default

#2-2
##model.setType(cv2.ml.SVM_NU_SVC)
##model.setNu(0.01)

#2-3
##model.setKernel(cv2.ml.SVM_LINEAR)

#2-4
##model.setKernel(cv2.ml.SVM_INTER) # histogram intersection

#2-5
##model.setKernel(cv2.ml.SVM_SIGMOID)
##model.setC(0.1)
##model.setGamma(0.00015)
##model.setCoef0(19.6)

#2-6
##model.setKernel(cv2.ml.SVM_POLY)
##model.setDegree(2)

#2-7
##model.setKernel(cv2.ml.SVM_RBF) #default
##model.setC(0.1)
##model.setGamma(0.00015)

#3
##crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
##model.setTermCriteria(crit)
##ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)

#4
ret = model.trainAuto(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
print("model.getType() = ", model.getType())
print("model.getKernelType()= ", model.getKernelType())
print("model.getC()= ", model.getC())
print("model.getNu()= ", model.getNu())
print("model.getGamma()= ", model.getGamma())
print("model.getDegree()= ", model.getDegree())
print("model.getCoef0()= ", model.getCoef0())

#5:
ret, y_pred = model.predict(x_train)
y_pred = y_pred.flatten()
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#6: x_test-> predictions -> pred
step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))

x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])

ret, pred = model.predict(x_test)
pred = pred.reshape(xx.shape)

#7: display data and result
#7-1
ax = plt.gca()
ax.set_aspect('equal')

#7-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#7-3
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                20, class_colors[label], 'o')
plt.show()
