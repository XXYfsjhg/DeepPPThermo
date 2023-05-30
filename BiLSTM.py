import numpy as np
import torch
from torch import nn, optim
from torch.utils import data as dataset
from SelfAttention import ScaledDotProductAttention
import random
import csv
from sklearn import metrics
import time

device = torch.device('cuda:0')
random_seed = random.randint(0, 100)
print(random_seed)


def csvread(csvname, length=2950):
    data_list = []
    with open(csvname, encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row_int = list(map(float, row[1:]))
            data_list.append(row_int)
    data_out = torch.tensor(data_list[0:length]).float()
    return data_out


def dataload2list(dataload):
    list_return = []
    for index, value in enumerate(dataload):
        list_return.append(value)
    return list_return


def datasplit(data0, split_rate=0.9, valid_rate=0.1):
    g = torch.Generator()
    g.manual_seed(random_seed)

    f = torch.Generator()
    f.manual_seed(random_seed)

    data_set = dataset.TensorDataset(data0)
    a = data_set.__len__()

    global valid_num
    valid_num = int(a * valid_rate)
    a = a - valid_num
    train_num = int(a * split_rate)
    test_num = a - int(a * split_rate)
    valid, b = dataset.random_split(data_set, [valid_num, a], generator=f)
    train, test = dataset.random_split(b, [train_num, test_num], generator=g)

    return train, test, valid, train_num, test_num


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        input_size = 32
        hidden_size = 128
        batch = 20
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                           bidirectional=True, dropout=0.4, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=1, bias=True)
        self.fc_out = nn.Linear(in_features=100, out_features=1, bias=True)
        self.sig = nn.Softsign()
        self.sigmoid = nn.Sigmoid()
        # self.sig = nn.ReLU()
        # dropout 以p概率归0， tensorflow中p是保留概率
        p = 0.3
        self.dropout = nn.Dropout(p)
        self.flatten = nn.Flatten(0, 1)
        self.sa = ScaledDotProductAttention(d_model=hidden_size * 2, d_k=hidden_size * 2, d_v=hidden_size * 2, h=6)
        self.sa1 = ScaledDotProductAttention(d_model=1, d_k=1, d_v=1, h=6)

        # self.attention = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)

    def forward(self, x):

        output, (hidden, cell) = self.rnn(x)

        sa = self.sa(output, output, output)
        sa = self.flatten(sa)
        sa = self.fc(sa)
        sa = self.sig(sa)
        sa = torch.reshape(sa, (-1, 100))

        out = self.sigmoid(sa)
        out = self.fc_out(out)
        out = self.sig(out)
        return out

'''
数据导入
'''

'''
data = torch.load("./Tensorzero_64_600.pt")
print(np.shape(data))
lendata = np.shape(data)[0]
print(np.shape(data), lendata)
len_h = int(lendata/2)
data_1 = data[0:len_h, :, :]
data_0 = data[len_h:, :, :]
'''
list_random = [333, 333, 444, 444, 555, 555, 666, 666, 777, 777]

data_LSTM_positive = torch.load('./Final_positive.pt')
data_LSTM_negative = torch.load('./Final_negative.pt')
data_LSTM_positive = data_LSTM_positive[0:2950, :, :]
data_LSTM_negative = data_LSTM_negative[0:2950, :, :]

for rand_seed in list_random:
    t1 = time.time()
    random_seed = random.randint(0, 100)
    print(random_seed)
    data_1_train, data_1_test, data_1_valid, train_num1, test_num1 = datasplit(data_LSTM_positive)
    data_0_train, data_0_test, data_0_valid, train_num0, test_num0 = datasplit(data_LSTM_negative)

    Iters = 119
    input_length = 100

    rnn = RNN()
    optimizer = optim.Adam(rnn.parameters(), lr=0.0001)  # sgd
    # optimizer = optim.SGD(rnn.parameters(), lr=1e-4, momentum=0.8)
    criteon = nn.MSELoss().to(device)
    # criteon = nn.BCELoss(weight=None, reduction='mean').to(device)  # 二进制交叉熵损失 BCELoss
    # criteon = nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None).to(device)
    rnn.to(device)
    rnn.train()

    print(valid_num)

    count1 = 0

    for epoch in range(50):  # 10iter   1min
        loss_sum = torch.tensor(0).float().view(1, 1, 1).to(device)
        train0 = dataset.DataLoader(data_0_train, batch_size=20, shuffle=True)
        train1 = dataset.DataLoader(data_1_train, batch_size=20, shuffle=True)
        train0 = dataload2list(train0)
        train1 = dataload2list(train1)

        for iteration in range(Iters):
            data = train1[iteration]
            x = data[0].to(device)

            y = torch.ones(20, 1).to(device)

            pred = rnn(x)
            loss = criteon(pred, y)
            loss_sum = loss_sum + loss
            optimizer.zero_grad()  # 清空过往梯度
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数

            data = train0[iteration]
            x = data[0].to(device)

            y = torch.zeros(20, 1).to(device)

            pred = rnn(x)
            loss = criteon(pred, y)
            loss_sum = loss_sum + loss
            optimizer.zero_grad()  # 清空过往梯度
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数

        print('迭代次数：', epoch, '当前迭代平均loss', loss_sum / (train_num0 + train_num1) * 20)

        # 计算训练结果
        n = 0
        m = 0
        rate = 0.0

        test0 = dataset.DataLoader(data_0_test, batch_size=1, shuffle=True)
        test1 = dataset.DataLoader(data_1_test, batch_size=1, shuffle=True)
        test0 = dataload2list(test0)
        test1 = dataload2list(test1)

        tensor_0 = torch.zeros(1, 1).to(device)
        tensor_1 = torch.ones(1, 1).to(device)

        for index in range(test_num0):

            data = test1[index]
            x = data[0].to(device)

            pred = rnn(x)
            pp = pred.detach()
            pp = torch.reshape(pp, (1, 1))
            loss0 = criteon(pp, tensor_0)
            loss1 = criteon(pp, tensor_1)
            if loss0 >= loss1:
                n = n + 1

        for index in range(test_num1):
            data = test0[index]
            x = data[0].to(device)

            pred = rnn(x)
            pp = pred.detach()
            pp = torch.reshape(pp, (1, 1))
            loss0 = criteon(pp, tensor_0)
            loss1 = criteon(pp, tensor_1)
            if loss0 < loss1:
                m = m + 1

        print(n, m)
        rate = (n + m) / (test_num0 + test_num1)
        print('rate_test', rate)
        print(test_num0, test_num1)
        if rate > 0.85:
            count1 = count1 + 1
            if count1 > 5:
                break

    # 计算训练结果
    # 计算训练结果
    n = 0
    m = 0
    rate = 0.0
    count = 0
    listindex = []

    valid0 = dataset.DataLoader(data_0_valid, batch_size=1, shuffle=True)
    valid1 = dataset.DataLoader(data_1_valid, batch_size=1, shuffle=True)
    valid0 = dataload2list(valid0)
    valid1 = dataload2list(valid1)

    tensor_0 = torch.zeros(1, 1).to(device)
    tensor_1 = torch.ones(1, 1).to(device)

    for index in range(valid_num):

        data = valid1[index]
        x = data[0].to(device)

        pred = rnn(x)
        pp = pred.detach()
        if count == 0:
            count = 1
            listpp = torch.clone(pp)
        else:
            listpp = torch.cat((listpp, pp), dim=0)
        listindex.append(1)

        pp = torch.reshape(pp, (1, 1))
        loss0 = criteon(pp, tensor_0)
        loss1 = criteon(pp, tensor_1)
        if loss0 >= loss1:
            n = n + 1

    for index in range(valid_num):
        data = valid0[index]
        x = data[0].to(device)

        pred = rnn(x)
        pp = pred.detach()
        listpp = torch.cat((listpp, pp), dim=0)
        listindex.append(0)
        pp = torch.reshape(pp, (1, 1))
        loss0 = criteon(pp, tensor_0)
        loss1 = criteon(pp, tensor_1)
        if loss0 < loss1:
            m = m + 1

    listpp = listpp.tolist()

    y = np.array(listindex)
    scores = np.array(listpp)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    print(auc)

    print(n, m)
    rate = (n + m) / (valid_num + valid_num)
    print('rate_valid', rate)

    t2 = time.time()
    print('时间:%s' % (t2 - t1))





