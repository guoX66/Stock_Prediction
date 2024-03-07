import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import figure
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import openpyxl as op
from LSTM import Seq2Seq, LSTM, CNN_LSTM


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def write_csv(data_list, file_name):
    wb = op.Workbook()
    ws = wb.create_sheet('Table', 0)
    for i in data_list:
        ws.append(i)
    ws_ = wb['Sheet']
    wb.remove(ws_)
    wb.save(f'{file_name}.xlsx')


def difference(data_set, interval=1):
    diff = [data_set[0]]
    for i in range(interval, len(data_set)):
        value = data_set[i] - data_set[i - interval]
        diff.append(value)
    return np.array(diff)


def row2col(y_total):
    data = []
    for j in range(len(y_total[0])):
        y_row = []
        for e_num in range(len(y_total)):
            y_row.append(y_total[e_num][j])
        data.append(y_row)
    data = np.array(data)
    return data


def invert_difference(first, yh):
    output = [first]
    for i in range(1, len(yh)):
        try:
            value = yh[i][0] + output[-1]
        except IndexError:
            value = yh[i] + output[-1]
        output.append(value)
    return np.array(output)


def process(need_col, data, batch_size, shuffle, interval, pred_size, pre=False):
    data = data.tolist()
    seq = []
    if not pre:
        for i in range(len(data) - interval - pred_size):
            train_seq = []
            train_label = []
            for j in range(i, i + interval):
                x = data[j]
                train_seq.append(x)
            for j in range(i + interval, i + interval + pred_size):
                x = [data[j][need_col]]
                train_label.append(x)
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
    else:
        for i in range(len(data) - interval + 1):
            train_seq = []
            train_label = []
            for j in range(i, i + interval):
                x = data[j]
                train_seq.append(x)
            for j in range(pred_size):
                x = [1]
                train_label.append(x)
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        # train_label.append(load[i + interval])

        # train_label = torch.FloatTensor(train_label)

    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq


def plot(fold_name, title, filename, y_pred_list, y_all, t, t_pred):
    figure(figsize=(12.8, 9.6))
    plt.plot(t_pred, y_pred_list, color='blue', label='预测曲线')
    plt.plot(t, y_all, color='red', label='实际曲线')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(f'{title}-训练结果', fontsize=20)
    plt.xlabel('距离首日天数', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f'{fold_name}/{title}-训练结果.png')
    plt.show()


def test_plot(fold_name, mode, title, lstm_model, t, Dte, filename, dataset, m, n, val_rate, device):
    lstm_model.eval()
    y_pred_list = []
    y_all = dataset

    print('predicting...')
    for seq, _ in Dte:
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = lstm_model(seq)
            if mode == 'std':
                y_pred = n * y_pred + m  # 标准化
            elif mode == 'maxmin':
                y_pred = (m - n) * y_pred + n  # 最大最小值归一化
            elif mode == 'total':
                y_pred = m * y_pred  # 整体归一化

            y_pred_list = y_pred_list + y_pred.tolist()[0]

    y_pred_list = np.array(y_pred_list)
    t_pred = t[len(y_pred[0]) + 1:len(t) - len(y_pred[0])]
    plot(fold_name, title, filename, y_pred_list, y_all, t, t_pred)


def train_test(pred_size, fold_name, title, dataset, filename, t, max_epochs, interval, val_rate, input_size,
               output_size,
               hidden_size,
               num_layers,
               batch_size, learn_rate, step_size, gamma, model_style, need_col):
    dataset = dataset.astype('float64')
    m_epoch = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU加速运算
    train = dataset[:int(len(dataset) * (1 - val_rate))]
    val = dataset[int(len(dataset) * (1 - val_rate)):]
    test = dataset
    Dtr = process(need_col, train, batch_size, True, interval, pred_size, pre=False)
    Val = process(need_col, val, batch_size, True, interval, pred_size, pre=False)
    if model_style == 0:
        lstm_model = LSTM(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                          batch_size=batch_size, device=device).to(device)
    elif model_style == 1:
        lstm_model = Seq2Seq(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                             batch_size=batch_size, device=device).to(device)
    elif model_style == 2:
        lstm_model = CNN_LSTM(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                              out_channels=32).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learn_rate)  # adam优化器
    loss_fn = nn.MSELoss()  # 损失函数，mse均值误差

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # 学习率调整器
    min_val_loss = np.Inf
    for epoch in tqdm(range(max_epochs)):  # 迭代次数
        train_loss = []
        lstm_model.train()
        for (seq, label) in Dtr:  # 数据集的数据seq和标签label
            seq = seq.to(device)
            seq = seq.half()
            label = label.to(device)
            y_pred = lstm_model(seq)
            loss = loss_fn(y_pred, label)  # 计算损失
            train_loss.append(loss.item())
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化更新权重

        scheduler.step()  # 学习率调整
        # 验证
        lstm_model.eval()  # 验证数据集时禁止反向传播优化权重
        total_val_loss = 0
        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            for data in Val:
                seq, label = data
                seq = seq.to(device)
                label = label.to(device)
                outputs = lstm_model(seq)
                loss = loss_fn(outputs, label)
                total_val_loss = total_val_loss + loss.item()
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            m_epoch = epoch
            torch.save(lstm_model.state_dict(), f"model-{filename}.pth")  # 保存最好的模型

    print()
    print(f'本次训练损失最小的epoch为{m_epoch},最小损失为{min_val_loss}')
    # 测试
    # model = torch.load(f"model-{filename}.pth")
    # test_plot(fold_name, mode, title, model, t, Dte, filename, dataset, m, n, val_rate, device, is_diff)


def LSTM_main(pred_size, fold_name, mode, title, t, data, max_epochs, interval, val_rate, input_size,
              output_size,
              hidden_size,
              num_layers,
              batch_size, learn_rate, step_size, gamma, model_style, need_col):
    cond = f'{need_col}'
    dataset = data

    ss_time = time.time()  # 开始时间
    train_test(pred_size, fold_name, title, dataset, cond, t, max_epochs, interval, val_rate, input_size,
               output_size,
               hidden_size,
               num_layers,
               batch_size, learn_rate, step_size, gamma, model_style, need_col)
    ed_time = time.time()  # 结束时间
    tt_time = ed_time - ss_time
    print("本次训练总共用时:{}小时:{}分钟:{}秒".format(int(tt_time // 3600), int((tt_time % 3600) // 60),
                                                       int(tt_time % 60)))
