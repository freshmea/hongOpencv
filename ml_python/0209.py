# 0209.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)

#1: load train data
with np.load('data/0201_data40.npz') as X: # '0201_data50.npz'
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.float32)
    height, width = X['size']

#2: EM clustering
#2-1: model.train()
model = cv2.ml.EM_create()
model.setClustersNumber(2)   # '0201_data40.npz'
##model.setClustersNumber(3) # '0201_data50.npz'
##model.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
##crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
##model.setTermCriteria(crit)

#2-2
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses= None)
means = model.getMeans()
print("means=", means)
covs = model.getCovs()
print("covs=", covs)
w = model.getWeights()
print("weights=", w)

#2-3: train using model.trainEM()
##K= model.getClustersNumber() # 2
##term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
##ret, labels, centers = cv2.kmeans(x_train, K, None, term_crit, 5,
##                                  cv2.KMEANS_RANDOM_CENTERS)
##print('centers.shape=', centers.shape)
##print('labels.shape=', labels.shape)
##print('ret=', ret)
##
####retval, logLikelihoods, labels, probs = model.trainE(samples=x_train, means0=centers)
####retval, logLikelihoods, labels, probs = model.trainM(samples=x_train, probs0= probs)
##retval, logLikelihoods, labels, probs = model.trainEM(samples=x_train)
##y_pred =  labels.flatten()
##
##means = model.getMeans()
##print("means=", means)
##covs = model.getCovs()
##print("covs=", covs)
##w = model.getWeights()
##print("weights=", w)

#2-4
ret, y_prob = model.predict(x_train) # y_prob.shape= (40, 2)
y_pred = np.argmax(y_prob, axis = 1) # y_pred.shape(40,)
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#2-5: x_test-> predictions -> pred
step = 2
xx, yy = np.meshgrid(np.arange(0, width,  step),
                     np.arange(0, height, step))

x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
ret, prob = model.predict(x_test) # prob.shape = (75000, 2)
pred = np.argmax(prob, axis = 1)
pred = pred.reshape(xx.shape)  # pred.shape = (250, 300)

#3: display data and result
#3-1
ax = plt.gca()
ax.set_aspect('equal')

#3-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = 'red', linewidths = 1)

#3-3
for label in range(2): #2 is number of class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                20, class_colors[label], 'o')
plt.show()
