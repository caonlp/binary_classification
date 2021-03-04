import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import codecs
from torch import nn
from torch import optim
import torch.nn.functional as F

def load_train_x():
    train_x = codecs.open('bert_dvd_train_uncls_aver.txt', mode = 'r', encoding = 'utf-8')
    line = train_x.readline()
    list1 = []
    while line:
        a = line.split()
        list1.append(a)
        line = train_x.readline()
    return np.array(list1)
    train_x.close()


def load_test_x():
    test_x = codecs.open('bert_dvd_test_uncls_aver.txt', mode = 'r', encoding = 'utf-8')
    line = test_x.readline()
    list1 = []
    while line:
        a = line.split()
        list1.append(a)
        line = test_x.readline()
    return np.array(list1)

def load_train_y():
    train_y = np.load('train_y.npy')
    return train_y
    train_y.close()

def load_test_y():
    test_y = np.load('test_y.npy')
    return test_y
    test_y.close()

train_x = load_train_x().astype(float)
test_x = load_test_x().astype(float)

train_y = load_train_y().astype(float)
test_y = load_test_y().astype(float)

max_epoch = 100
train_size = train_x.shape[0]
batch_size = 10
n_batch = train_size // batch_size

class DealTrainDataSet(Dataset):
    def __init__(self):
        self.train_data = torch.from_numpy(train_x)
        self.train_target = torch.from_numpy(train_y)
        self.len = train_x.shape[0]

    def __getitem__(self, index):
        return self.train_data[index], self.train_target[index]

    def __len__(self):
        return self.len

class DealTestDataSet(Dataset):
    def __init__(self):
        self.test_data = torch.from_numpy(test_x)
        self.test_target = torch.from_numpy(test_y)
        self.len = test_x.shape[0]

    def __getitem__(self, index):
        return self.test_data[index], self.test_target[index]

    def __len__(self):
        return self.len

dealTrainDataSet = DealTrainDataSet()
train_loader = DataLoader(dataset = dealTrainDataSet, batch_size = batch_size, shuffle = True)

dealTestDataSet = DealTestDataSet()
test_loader = DataLoader(dataset = dealTestDataSet, batch_size = batch_size, shuffle = False)

class MyModel(nn.Module):
    """
    定义了一个简单的三层全连接神经网络,每一层都是线性的
    """
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, output_dim):
        super(MyModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_layer_1)
        self.layer_2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.layer_3 = nn.Linear(hidden_layer_2, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

class MyModelWithActivation(nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, output_dim):
        super(MyModelWithActivation, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(input_dim, hidden_layer_1), nn.ReLU(True))
        self.layer_2 = nn.Sequential(nn.Linear(hidden_layer_1, hidden_layer_2), nn.ReLU(True))
        self.layer_3 = nn.Sequential(nn.Linear(hidden_layer_2, output_dim))
        """
        这里的Sequential（）函数的功能是将网络的层组合到一起
        """

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

class MyModelWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, output_dim):
        super(MyModelWithBatchNorm, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(input_dim, hidden_layer_1), nn.BatchNorm1d(hidden_layer_1), nn.ReLU(True))
        self.layer_2 = nn.Sequential(nn.Linear(hidden_layer_1, hidden_layer_2), nn.BatchNorm1d(hidden_layer_2), nn.ReLU(True))
        self.layer_3 = nn.Sequential(nn.Linear(hidden_layer_2, output_dim))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

classes = ('0', '1')

def train_process():
    # 神经网络结构
    model = MyModelWithActivation(768, 200, 50, 2)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    cost = nn.CrossEntropyLoss()
    # train
    for epoch in range(max_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print("[%d %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    print("Finished Training")
    # 保存模型
    torch.save(model, "mymodel.pkl") # 保存整个神经网络的结构和模型参数
    torch.save(model.state_dict(), 'model_params.pkl') # 只保存神经网络的模型参数

def reload_model():
    trained_model = torch.load('mymodel.pkl')
    return trained_model


def test_process():
    model = reload_model()
    dataiter = iter(test_loader)
    test_data, test_target = dataiter.next()
    test_data, test_target = test_data.float(), test_target.long()
    outputs = model(Variable(test_data))
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    print(test_y[:10])


def test():
    test_loss = 0
    correct = 0
    model = reload_model()
    for test_data, test_target in test_loader:
        test_data, test_target = test_data.float(), test_target.long()
        test_data, test_target = Variable(test_data), Variable(test_target)
        outputs = model(Variable(test_data))
        # sum up batch loss
        test_loss += F.nll_loss(outputs, test_target, reduction='sum').item()
        pred = outputs.data.max(1, keepdim = True)[1]
        correct += pred.eq(test_target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    train_process()
    reload_model()
    test()