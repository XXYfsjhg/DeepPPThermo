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

list_random = [333, 333, 444, 444, 555, 555, 666, 666, 777, 777]


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
    f.manual_seed(rand_seed)

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
    def __init__(self, length_feature1, length_feature2, length_feature3, length_feature4, length_feature5, length_feature6):
        super(RNN, self).__init__()
        input_size = 32
        hidden_size = 64
        batch = 20
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                           bidirectional=True, dropout=0.4, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=1, bias=True)
        self.fc1 = nn.Linear(in_features=length_feature1, out_features=100, bias=True)
        self.fc2 = nn.Linear(in_features=length_feature2, out_features=100, bias=True)
        self.fc3 = nn.Linear(in_features=length_feature3, out_features=100, bias=True)
        self.fc4 = nn.Linear(in_features=length_feature4, out_features=100, bias=True)
        self.fc5 = nn.Linear(in_features=length_feature5, out_features=100, bias=True)
        self.fc6 = nn.Linear(in_features=length_feature6, out_features=100, bias=True)
        self.fc_sa = nn.Linear(in_features=600, out_features=100, bias=True)

        self.fc_out = nn.Linear(in_features=700, out_features=1, bias=True)
        self.sig = nn.Softsign()
        self.sigmoid = nn.Sigmoid()
        # self.sig = nn.ReLU()
        # dropout 以p概率归0， tensorflow中p是保留概率
        p = 0.3
        self.dropout = nn.Dropout(p)
        self.flatten = nn.Flatten(0, 1)
        self.sa = ScaledDotProductAttention(d_model=hidden_size * 2, d_k=hidden_size * 2, d_v=hidden_size * 2, h=5)
        self.sa1 = ScaledDotProductAttention(d_model=1, d_k=1, d_v=1, h=5)

        # self.attention = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)

    def forward(self, x, x1, x2, x3, x4, x5, x6):

        output, (hidden, cell) = self.rnn(x)

        sa = self.sa(output, output, output)
        sa = self.flatten(sa)
        sa = self.fc(sa)
        sa = self.sig(sa)
        sa = torch.reshape(sa, (-1, 100))

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)
        x5 = self.fc5(x5)
        x6 = self.fc6(x6)

        x1 = torch.unsqueeze(x1, 2)
        x2 = torch.unsqueeze(x2, 2)
        x3 = torch.unsqueeze(x3, 2)
        x4 = torch.unsqueeze(x4, 2)
        x5 = torch.unsqueeze(x5, 2)
        x6 = torch.unsqueeze(x6, 2)
        sa = torch.unsqueeze(sa, 2)

        x_sa = torch.cat((sa, x1, x2, x3, x4, x5, x6), dim=1)
        x_sa = self.sa1(x_sa, x_sa, x_sa)
        out = torch.squeeze(x_sa)
        out = self.sigmoid(out)
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

data_LSTM_positive = torch.load('./Final_positive.pt')
data_LSTM_negative = torch.load('./Final_negative.pt')
data_LSTM_negative = data_LSTM_negative[0:2950, :, :]
data_LSTM_positive = data_LSTM_positive[0:2950, :, :]

print(np.shape(data_LSTM_positive), np.shape(data_LSTM_positive))

negative_csv_DPC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\negativePAAC.csv'
positive_csv_DPC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\PAAC.csv'

negative_csv_CKSAAP = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\negativeCTriad.csv'
positive_csv_CKSAAP = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\CTriad.csv'

negative_csv_CTriad = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\negativeCKSAAP.csv'
positive_csv_CTriad = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\CKSAAP.csv'

negative_csv_ASDC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\negativeDPC.csv'
positive_csv_ASDC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\DPC.csv'

negative_csv_AAC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\negativeAPAAC.csv'
positive_csv_AAC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\APAAC.csv'

negative_csv_PAAC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\negativeAAC.csv'
positive_csv_PAAC = 'D:\\pro_design\\doc2vec+lstm\\doc2vec+lstm\\input\\csv\\AAC.csv'

data_negative_DPC = csvread(negative_csv_DPC)
data_positive_DPC = csvread(positive_csv_DPC)

data_negative_CKSAAP = csvread(negative_csv_CKSAAP)
data_positive_CKSAAP = csvread(positive_csv_CKSAAP)

data_negative_CTriad = csvread(negative_csv_CTriad)
data_positive_CTriad = csvread(positive_csv_CTriad)

data_negative_ASDC = csvread(negative_csv_ASDC)
data_positive_ASDC = csvread(positive_csv_ASDC)

data_negative_AAC = csvread(negative_csv_AAC)
data_positive_AAC = csvread(positive_csv_AAC)

data_negative_PAAC = csvread(negative_csv_PAAC)
data_positive_PAAC = csvread(positive_csv_PAAC)

for rand_seed in list_random:
    t1 = time.time()
    random_seed = random.randint(0, 100)
    print(random_seed)

    data_1_train, data_1_test, data_1_valid, train_num1, test_num1 = datasplit(data_LSTM_positive)
    data_0_train, data_0_test, data_0_valid, train_num0, test_num0 = datasplit(data_LSTM_negative)

    data_CKSAAP_train_1, data_CKSAAP_test_1, data_CKSAAP_valid_1, _, _ = datasplit(data_positive_CKSAAP)
    data_CKSAAP_train_0, data_CKSAAP_test_0, data_CKSAAP_valid_0, _, _ = datasplit(data_negative_CKSAAP)

    data_DPC_train_1, data_DPC_test_1, data_DPC_valid_1, _, _ = datasplit(data_positive_DPC)
    data_DPC_train_0, data_DPC_test_0, data_DPC_valid_0, _, _ = datasplit(data_negative_DPC)

    data_CTriad_train_1, data_CTriad_test_1, data_CTriad_valid_1, _, _ = datasplit(data_positive_CTriad)
    data_CTriad_train_0, data_CTriad_test_0, data_CTriad_valid_0, _, _ = datasplit(data_negative_CTriad)

    data_ASDC_train_1, data_ASDC_test_1, data_ASDC_valid_1, _, _ = datasplit(data_positive_ASDC)
    data_ASDC_train_0, data_ASDC_test_0, data_ASDC_valid_0, _, _ = datasplit(data_negative_ASDC)

    data_AAC_train_1, data_AAC_test_1, data_AAC_valid_1, _, _ = datasplit(data_positive_AAC)
    data_AAC_train_0, data_AAC_test_0, data_AAC_valid_0, _, _ = datasplit(data_negative_AAC)

    data_PAAC_train_1, data_PAAC_test_1, data_PAAC_valid_1, _, _ = datasplit(data_positive_PAAC)
    data_PAAC_train_0, data_PAAC_test_0, data_PAAC_valid_0, _, _ = datasplit(data_negative_PAAC)

    length_feature1 = np.shape(data_negative_DPC)[1]
    length_feature2 = np.shape(data_negative_CKSAAP)[1]
    length_feature3 = np.shape(data_negative_CTriad)[1]
    length_feature4 = np.shape(data_negative_ASDC)[1]
    length_feature5 = np.shape(data_negative_AAC)[1]
    length_feature6 = np.shape(data_negative_PAAC)[1]

    Iters = 119
    input_length = 100

    rnn = RNN(length_feature1, length_feature2, length_feature3, length_feature4, length_feature5, length_feature6)
    optimizer = optim.Adam(rnn.parameters(), lr=1e-4)  # sgd
    # optimizer = optim.SGD(rnn.parameters(), lr=1e-4, momentum=0.8)
    criteon = nn.MSELoss().to(device)
    # criteon = nn.BCELoss(weight=None, reduction='mean').to(device)  # 二进制交叉熵损失 BCELoss
    # criteon = nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None).to(device)
    rnn.to(device)
    rnn.train()
    batch_size = 10

    print(valid_num)

    for epoch in range(100):  # 10iter   1min
        loss_sum = torch.tensor(0).float().view(1, 1, 1).to(device)
        train0 = dataset.DataLoader(data_0_train, batch_size=batch_size, shuffle=True)
        train1 = dataset.DataLoader(data_1_train, batch_size=batch_size, shuffle=True)
        train0 = dataload2list(train0)
        train1 = dataload2list(train1)

        train_DPC_0 = dataset.DataLoader(data_DPC_train_0, batch_size=batch_size, shuffle=True)
        train_DPC_1 = dataset.DataLoader(data_DPC_train_1, batch_size=batch_size, shuffle=True)
        train_DPC_0 = dataload2list(train_DPC_0)
        train_DPC_1 = dataload2list(train_DPC_1)

        train_CKSAAP_0 = dataset.DataLoader(data_CKSAAP_train_0, batch_size=batch_size, shuffle=True)
        train_CKSAAP_1 = dataset.DataLoader(data_CKSAAP_train_1, batch_size=batch_size, shuffle=True)
        train_CKSAAP_0 = dataload2list(train_CKSAAP_0)
        train_CKSAAP_1 = dataload2list(train_CKSAAP_1)

        train_CTriad_0 = dataset.DataLoader(data_CTriad_train_0, batch_size=batch_size, shuffle=True)
        train_CTriad_1 = dataset.DataLoader(data_CTriad_train_1, batch_size=batch_size, shuffle=True)
        train_CTriad_0 = dataload2list(train_CTriad_0)
        train_CTriad_1 = dataload2list(train_CTriad_1)

        train_ASDC_0 = dataset.DataLoader(data_ASDC_train_0, batch_size=batch_size, shuffle=True)
        train_ASDC_1 = dataset.DataLoader(data_ASDC_train_1, batch_size=batch_size, shuffle=True)
        train_ASDC_0 = dataload2list(train_ASDC_0)
        train_ASDC_1 = dataload2list(train_ASDC_1)

        train_AAC_0 = dataset.DataLoader(data_AAC_train_0, batch_size=batch_size, shuffle=True)
        train_AAC_1 = dataset.DataLoader(data_AAC_train_1, batch_size=batch_size, shuffle=True)
        train_AAC_0 = dataload2list(train_AAC_0)
        train_AAC_1 = dataload2list(train_AAC_1)

        train_PAAC_0 = dataset.DataLoader(data_PAAC_train_0, batch_size=batch_size, shuffle=True)
        train_PAAC_1 = dataset.DataLoader(data_PAAC_train_1, batch_size=batch_size, shuffle=True)
        train_PAAC_0 = dataload2list(train_PAAC_0)
        train_PAAC_1 = dataload2list(train_PAAC_1)

        for iteration in range(Iters):
            data = train1[iteration]
            x = data[0].to(device)

            data_DPC = train_DPC_1[iteration]
            x_DPC = data_DPC[0].to(device)

            data_CKSAAP = train_CKSAAP_1[iteration]
            x_CKSAAP = data_CKSAAP[0].to(device)

            data_CTriad = train_CTriad_1[iteration]
            x_CTriad = data_CTriad[0].to(device)

            data_ASDC = train_ASDC_1[iteration]
            x_ASDC = data_ASDC[0].to(device)

            data_AAC = train_AAC_1[iteration]
            x_AAC = data_AAC[0].to(device)

            data_PAAC = train_PAAC_1[iteration]
            x_PAAC = data_PAAC[0].to(device)

            y = torch.ones(batch_size, 1).to(device)

            pred = rnn(x, x_DPC, x_CKSAAP, x_CTriad, x_ASDC, x_AAC, x_PAAC)
            loss = criteon(pred, y)
            loss_sum = loss_sum + loss
            optimizer.zero_grad()  # 清空过往梯度
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数

            data = train0[iteration]
            x = data[0].to(device)

            data_DPC = train_DPC_0[iteration]
            x_DPC = data_DPC[0].to(device)

            data_CKSAAP = train_CKSAAP_0[iteration]
            x_CKSAAP = data_CKSAAP[0].to(device)

            data_CTriad = train_CTriad_0[iteration]
            x_CTriad = data_CTriad[0].to(device)

            data_ASDC = train_ASDC_0[iteration]
            x_ASDC = data_ASDC[0].to(device)

            data_AAC = train_AAC_0[iteration]
            x_AAC = data_AAC[0].to(device)

            data_PAAC = train_PAAC_0[iteration]
            x_PAAC = data_PAAC[0].to(device)

            y = torch.zeros(batch_size, 1).to(device)

            pred = rnn(x, x_DPC, x_CKSAAP, x_CTriad, x_ASDC, x_AAC, x_PAAC)
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

        test_DPC_0 = dataset.DataLoader(data_DPC_test_0, batch_size=1, shuffle=True)
        test_DPC_1 = dataset.DataLoader(data_DPC_test_1, batch_size=1, shuffle=True)
        test_DPC_0 = dataload2list(test_DPC_0)
        test_DPC_1 = dataload2list(test_DPC_1)

        test_CKSAAP_0 = dataset.DataLoader(data_CKSAAP_test_0, batch_size=1, shuffle=True)
        test_CKSAAP_1 = dataset.DataLoader(data_CKSAAP_test_1, batch_size=1, shuffle=True)
        test_CKSAAP_0 = dataload2list(test_CKSAAP_0)
        test_CKSAAP_1 = dataload2list(test_CKSAAP_1)

        test_CTriad_0 = dataset.DataLoader(data_CTriad_test_0, batch_size=1, shuffle=True)
        test_CTriad_1 = dataset.DataLoader(data_CTriad_test_1, batch_size=1, shuffle=True)
        test_CTriad_0 = dataload2list(test_CTriad_0)
        test_CTriad_1 = dataload2list(test_CTriad_1)

        test_ASDC_0 = dataset.DataLoader(data_ASDC_test_0, batch_size=1, shuffle=True)
        test_ASDC_1 = dataset.DataLoader(data_ASDC_test_1, batch_size=1, shuffle=True)
        test_ASDC_0 = dataload2list(test_ASDC_0)
        test_ASDC_1 = dataload2list(test_ASDC_1)

        test_AAC_0 = dataset.DataLoader(data_AAC_test_0, batch_size=1, shuffle=True)
        test_AAC_1 = dataset.DataLoader(data_AAC_test_1, batch_size=1, shuffle=True)
        test_AAC_0 = dataload2list(test_AAC_0)
        test_AAC_1 = dataload2list(test_AAC_1)

        test_PAAC_0 = dataset.DataLoader(data_PAAC_test_0, batch_size=1, shuffle=True)
        test_PAAC_1 = dataset.DataLoader(data_PAAC_test_1, batch_size=1, shuffle=True)
        test_PAAC_0 = dataload2list(test_PAAC_0)
        test_PAAC_1 = dataload2list(test_PAAC_1)

        tensor_0 = torch.zeros(1, 1).to(device)
        tensor_1 = torch.ones(1, 1).to(device)

        for index in range(test_num0):

            data = test1[index]
            x = data[0].to(device)

            data_DPC = test_DPC_1[index]
            x_DPC = data_DPC[0].to(device)

            data_CKSAAP = test_CKSAAP_1[index]
            x_CKSAAP = data_CKSAAP[0].to(device)

            data_CTriad = test_CTriad_1[index]
            x_CTriad = data_CTriad[0].to(device)

            data_ASDC = test_ASDC_1[index]
            x_ASDC = data_ASDC[0].to(device)

            data_AAC = test_AAC_1[index]
            x_AAC = data_AAC[0].to(device)

            data_PAAC = test_PAAC_1[index]
            x_PAAC = data_PAAC[0].to(device)

            pred = rnn(x, x_DPC, x_CKSAAP, x_CTriad, x_ASDC, x_AAC, x_PAAC)
            pp = pred.detach()
            pp = torch.reshape(pp, (1, 1))
            loss0 = criteon(pp, tensor_0)
            loss1 = criteon(pp, tensor_1)
            if loss0 >= loss1:
                n = n + 1

        for index in range(test_num1):
            data = test0[index]
            x = data[0].to(device)

            data_DPC = test_DPC_0[index]
            x_DPC = data_DPC[0].to(device)

            data_CKSAAP = test_CKSAAP_0[index]
            x_CKSAAP = data_CKSAAP[0].to(device)

            data_CTriad = test_CTriad_0[index]
            x_CTriad = data_CTriad[0].to(device)

            data_ASDC = test_ASDC_0[index]
            x_ASDC = data_ASDC[0].to(device)

            data_AAC = test_AAC_0[index]
            x_AAC = data_AAC[0].to(device)

            data_PAAC = test_PAAC_0[index]
            x_PAAC = data_PAAC[0].to(device)

            pred = rnn(x, x_DPC, x_CKSAAP, x_CTriad, x_ASDC, x_AAC, x_PAAC)
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

    valid_DPC_0 = dataset.DataLoader(data_DPC_valid_0, batch_size=1, shuffle=True)
    valid_DPC_1 = dataset.DataLoader(data_DPC_valid_1, batch_size=1, shuffle=True)
    valid_DPC_0 = dataload2list(valid_DPC_0)
    valid_DPC_1 = dataload2list(valid_DPC_1)

    valid_CKSAAP_0 = dataset.DataLoader(data_CKSAAP_valid_0, batch_size=1, shuffle=True)
    valid_CKSAAP_1 = dataset.DataLoader(data_CKSAAP_valid_1, batch_size=1, shuffle=True)
    valid_CKSAAP_0 = dataload2list(valid_CKSAAP_0)
    valid_CKSAAP_1 = dataload2list(valid_CKSAAP_1)

    valid_CTriad_0 = dataset.DataLoader(data_CTriad_valid_0, batch_size=1, shuffle=True)
    valid_CTriad_1 = dataset.DataLoader(data_CTriad_valid_1, batch_size=1, shuffle=True)
    valid_CTriad_0 = dataload2list(valid_CTriad_0)
    valid_CTriad_1 = dataload2list(valid_CTriad_1)

    valid_ASDC_0 = dataset.DataLoader(data_ASDC_valid_0, batch_size=1, shuffle=True)
    valid_ASDC_1 = dataset.DataLoader(data_ASDC_valid_1, batch_size=1, shuffle=True)
    valid_ASDC_0 = dataload2list(valid_ASDC_0)
    valid_ASDC_1 = dataload2list(valid_ASDC_1)

    valid_AAC_0 = dataset.DataLoader(data_AAC_valid_0, batch_size=1, shuffle=True)
    valid_AAC_1 = dataset.DataLoader(data_AAC_valid_1, batch_size=1, shuffle=True)
    valid_AAC_0 = dataload2list(valid_AAC_0)
    valid_AAC_1 = dataload2list(valid_AAC_1)

    valid_PAAC_0 = dataset.DataLoader(data_PAAC_valid_0, batch_size=1, shuffle=True)
    valid_PAAC_1 = dataset.DataLoader(data_PAAC_valid_1, batch_size=1, shuffle=True)
    valid_PAAC_0 = dataload2list(valid_PAAC_0)
    valid_PAAC_1 = dataload2list(valid_PAAC_1)

    tensor_0 = torch.zeros(1, 1).to(device)
    tensor_1 = torch.ones(1, 1).to(device)

    for index in range(valid_num):

        data = valid1[index]
        x = data[0].to(device)

        data_DPC = valid_DPC_1[index]
        x_DPC = data_DPC[0].to(device)

        data_CKSAAP = valid_CKSAAP_1[index]
        x_CKSAAP = data_CKSAAP[0].to(device)

        data_CTriad = valid_CTriad_1[index]
        x_CTriad = data_CTriad[0].to(device)

        data_ASDC = valid_ASDC_1[index]
        x_ASDC = data_ASDC[0].to(device)

        data_AAC = valid_AAC_1[index]
        x_AAC = data_AAC[0].to(device)

        data_PAAC = valid_PAAC_1[index]
        x_PAAC = data_PAAC[0].to(device)

        pred = rnn(x, x_DPC, x_CKSAAP, x_CTriad, x_ASDC, x_AAC, x_PAAC)
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

        data_DPC = valid_DPC_0[index]
        x_DPC = data_DPC[0].to(device)

        data_CKSAAP = valid_CKSAAP_0[index]
        x_CKSAAP = data_CKSAAP[0].to(device)

        data_CTriad = valid_CTriad_0[index]
        x_CTriad = data_CTriad[0].to(device)

        data_ASDC = valid_ASDC_0[index]
        x_ASDC = data_ASDC[0].to(device)

        data_AAC = valid_AAC_0[index]
        x_AAC = data_AAC[0].to(device)

        data_PAAC = valid_PAAC_0[index]
        x_PAAC = data_PAAC[0].to(device)

        pred = rnn(x, x_DPC, x_CKSAAP, x_CTriad, x_ASDC, x_AAC, x_PAAC)
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







