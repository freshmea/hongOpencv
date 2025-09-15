#0410.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary  # pip install torchsummary


#1-1
def load_Iris():
    label = {'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("data/iris.csv", skiprows = 1, delimiter = ',',
                      converters={4: lambda name: label[name.decode()]})
    return np.float32(data)
iris_data = load_Iris()

np.random.seed(1)
def train_test_split(iris_data, ratio = 0.8, shuffle = True): # train: 0.8, test: 0.2
    if shuffle:
        np.random.shuffle(iris_data)

    n = int(iris_data.shape[0] * ratio)
    x_train = iris_data[:n, :-1]
    y_train = iris_data[:n, -1]

    x_test = iris_data[n:, :-1]
    y_test = iris_data[n:, -1]
    return (x_train, y_train), (x_test, y_test)

# pip install scikit-learn
# from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = train_test_split(iris_data)

print(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
print(f"x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")

#1-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_train_1hot = torch.nn.functional.one_hot(y_train.long(), num_classes=3).float()

x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

ds = TensorDataset(x_train, y_train_1hot)
loader = DataLoader(dataset=ds, batch_size=32,  shuffle=True)


#2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features=4, out_features=10, bias=True)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        #x = torch.sigmoid(x) # with MSELoss()
        return x

model = Net().to(DEVICE)
summary(model, x_train.shape, device=DEVICE)

#3-1
#loss_fn = torch.nn.MSELoss()
loss_fn  = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#3-2
loss_list = []
iter_per_epoch = int(np.ceil(x_train.shape[0]//loader.batch_size)) # 120//32 = 4
model.train()
for epoch in range(100):

    batch_loss = 0.0
    for X, y in loader:
        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()*X.size(0) # mean -> sum

    batch_loss /= x_train.shape[0] # divide by train size
    print("epoch=", epoch, "batch_loss=", batch_loss)
    loss_list.append(batch_loss)

#3-3
from matplotlib import pyplot as plt

plt.plot(loss_list)
plt.show()

#3-4
model.eval()
with torch.no_grad():

    out = model(x_train)
    pred = torch.max(out.data, 1)[1]

    accuracy = (pred == y_train).float().mean() # train accuracy
    #accuracy = pred.eq(y_train.view_as(pred)).float().mean()
    print("train accuracy= ", accuracy.item())

    out = model(x_test)
    pred = torch.max(out.data, 1)[1]
    accuracy = (pred == y_test).float().mean() # test accuracy
    print("test accuracy= ", accuracy.item())

#4
dummy_input = torch.randn(120, 4).to(DEVICE)  # x_train
torch.onnx.export(model, dummy_input, "./dnn/IRIS.onnx")
