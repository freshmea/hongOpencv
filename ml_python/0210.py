#0210.py
import matplotlib.pyplot as plt
import numpy as np

#1: param1 = 1, param2 = 1, cv2.ml.ANN_MLP_SIGMOID_SYM
alpha = 1
beta  = 1
x = np.linspace(-10, 10, num=100)
y = beta*(1-np.exp(-alpha*x))/(1+np.exp(-alpha*x))
plt.plot(x, y, label='alpha=1, beta=1')

#2: param1 = 0, param2 = 0, cv2.ml.ANN_MLP_SIGMOID_SYM
alpha = 2/3
beta  = 1.7159
y = beta*(1-np.exp(-alpha*x))/(1+np.exp(-alpha*x))
plt.plot(x, y, label='alpha=2/3, beta=1.7159')

#3
###y = np.tanh(x)
##y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
##plt.plot(x, y, label='tanh')
##
#4
##y = beta*np.tanh(x*alpha)
##plt.plot(x, y, label='beta*np.tanh(x*alpha)')
plt.legend(loc='best')
plt.show()
