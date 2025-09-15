#0409.py
# pip install torchsummary
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # pip install torchsummary

import onnxruntime as ort  # pip install onnxruntime

#1
#1-1
x_train = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype = np.float32)

##y_train = np.array([0,0,0,1], dtype = np.int32)   # AND
##y_train = np.array([0,1,1,1], dtype = np.float32) # OR
y_train = np.array([0,1,1,0], dtype = np.float32)   # XOR

#1-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train = torch.tensor(x_train).to(DEVICE)
y =       torch.tensor(y_train).to(DEVICE)  # category: 0, 1

y_train = torch.nn.functional.one_hot(y.long(), num_classes=2).float()   # one-hot

#2
model = nn.Sequential(nn.Linear(in_features=2, out_features=4, bias=True),
                      nn.Sigmoid(),
                      nn.Linear(4, 2),
                      nn.Softmax(dim=1)
                      ).to(DEVICE)

summary(model, (4, 2), device=DEVICE)
print('model = ', model)

#3-1
loss_fn  = torch.nn.MSELoss()
##def loss_fn(output, target):
##    loss = torch.mean((output - target)**2)
##    return loss

#loss_fn  = torch.nn.CrossEntropyLoss()
##def softmax(x):
##    exp_x = torch.exp(x)
##    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
##    return exp_x/sum_x

##def loss_fn(output, target): # NLL(Negative Log Likelihood)
##    y = torch.argmax(target, dim=1)
##    log_softmax = torch.nn.LogSoftmax(dim=1)(output)
##    return torch.nn.NLLLoss()(log_softmax, y)
##    #return F.nll_loss(F.log_softmax(output, dim=1), y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

#3-2
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(x_train)

    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    #if epoch%10 == 0:
    #    print(epoch, loss.item())

##params = list(model.parameters())
##print('params = ', params)

#3-3
model.eval() # model.train(mode=False)
with torch.no_grad():
    out = model(x_train)
    pred= torch.max(out.data, dim=1)[1]

    accuracy = (pred == y).float().mean()
    #accuracy = pred.eq(y.view_as(pred)).float().mean()
    print("accuracy= ", accuracy.item())

#4
dummy_input = torch.randn(4, 2).to(DEVICE) # x_train
##torch.onnx.export(model, dummy_input, "./dnn/AND.onnx")
##torch.onnx.export(model, dummy_input, "./dnn/OR.onnx")
torch.onnx.export(model, dummy_input, "./dnn/XOR.onnx")

#5: check model
onnx_model=onnx.load("./dnn/XOR.onnx")
onnx.checker.check_model(onnx_model)
print("model checked!")

#6
ort_sess = ort.InferenceSession("./dnn/XOR.onnx")

X = x_train.cpu().numpy()
ort_inputs = {ort_sess.get_inputs()[0].name: X}
ort_outs = ort_sess.run(None, ort_inputs)[0]
ort_pred = np.argmax(ort_outs, axis = 1)
print('ort_pred=', ort_pred)

#7: save and load PyTorch model
torch.save(model,  "./dnn/XOR.pt")
model = torch.load( "./dnn/XOR.pt", weights_only=False)
model.eval()
with torch.no_grad():
    out = model(x_train)
    pred= torch.max(out.data, dim=1)[1]
print('pred=', pred.cpu().numpy())
