#0211.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)

#1
x_train = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]], dtype = np.float32)

y_train = np.array([0,0,0,1], dtype = np.float32)  # AND
##y_train = np.array([0,1,1,1], dtype = np.float32) # OR

#2: Artificial Neural Networks
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(np.array([2, 1]))
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

##model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)        # [-1.7, 1.7]
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1) # [-1, 1]
model.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                       cv2.TERM_CRITERIA_COUNT,1000,1e-5))

#3: train
#3-1
# 데이터 정제!
# 데이터로더 -> 배치사이즈 batch -> epoch(전체 데이터 1회양) -> shuffle -> 100 epoch
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)

#3-2: weights
layerSize = model.getLayerSizes()
for i in range(layerSize.shape[0] + 2):
    print("weights[{}] = {}".format(i, model.getWeights(i)))

#4
ret, y_out = model.predict(x_train) # infer. 순전파.
print("y_out=", y_out)
##y_pred = np.round(y_out)
y_pred = np.int32(y_out>0.5)
y_pred = y_pred.flatten()
print("y_pred=", y_pred)
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#5
h = 0.01
xx, yy = np.meshgrid(np.arange(0-2*h, 1+2*h, h),
                     np.arange(0-2*h, 1+2*h, h))

sample = np.c_[xx.ravel(), yy.ravel()]
ret, out = model.predict(sample)
pred = np.int32(out > 0.5)
pred = pred.reshape(xx.shape)

#6
ax = plt.gca()
ax.set_aspect('equal')

plt.contourf(xx, yy, pred, cmap=plt.cm.gray)
plt.contour(xx, yy, pred, colors='red', linewidths=1)

class_colors = ['blue', 'red']
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                50, class_colors[label], 'o')

plt.show()
