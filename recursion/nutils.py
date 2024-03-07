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


def row2col(y_total):
    data = []
    for j in range(len(y_total[0])):
        y_row = []
        for e_num in range(len(y_total)):
            y_row.append(y_total[e_num][j])
        data.append(y_row)
    data = np.array(data)
    return data


def difference(data_set, interval=1):
    diff = [data_set[0]]
    for i in range(interval, len(data_set)):
        value = data_set[i] - data_set[i - interval]
        diff.append(value)
    return np.array(diff)


def invert_difference(first, yh):
    output = [first]
    for i in range(1, len(yh)):
        try:
            value = yh[i][0] + output[-1]
        except:
            value = yh[i] + output[-1]
        output.append(value)
    return np.array(output)


def process(data, batch_size, shuffle, interval, pre=False):
    load = data
    try:
        load = load.tolist()
    except:
        pass
    seq = []
    if not pre:
        for i in range(len(data) - interval):
            train_seq = []
            train_label = []
            for j in range(i, i + interval):
                x = load[j]
                train_seq.append(x)
            train_label.append(load[i + interval])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
    else:
        for i in range(len(data) - interval + 1):
            train_seq = []
            train_label = []
            for j in range(i, i + interval):
                x = load[j]
                train_seq.append(x)
            train_label.append(1)
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq


def plot(fold_name, title, filename, y_pred_list, y_all, t_for_testing, t):
    figure(figsize=(12.8, 9.6))
    plt.plot(t_for_testing, y_pred_list, color='blue', label='预测曲线')
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


def test_plot(fold_name, mode, title_list, lstm_model, t, Dte, filename, dataset, mn_list, device):
    lstm_model.eval()
    y_pred_list = []
    y_all = dataset

    print('predicting...')
    for seq, _ in Dte:
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = lstm_model(seq)
            y_pred_list += y_pred.tolist()

    y_pred_list = np.array(y_pred_list)
    t_for_testing = t[len(seq[0]):len(seq[0]) + len(y_pred_list)]
    y_pred_list = row2col(y_pred_list)

    for i in range(len(title_list)):
        title = title_list[i]
        single_y_pred = y_pred_list[i]
        single_y_all = y_all[i]
        m, n = mn_list[i]
        if mode == 'std':
            single_y_pred = n * single_y_pred + m
        elif mode == 'maxmin':
            single_y_pred = (m - n) * single_y_pred + n
        elif mode == 'total':
            single_y_pred = m * single_y_pred
        plot(fold_name, title, filename, single_y_pred, single_y_all, t_for_testing, t)


def train_test(fold_name, mode, title_list, dataset, filename, t, max_epochs, interval, val_rate, input_size,
               output_size,
               hidden_size,
               num_layers,
               batch_size, learn_rate, step_size, gamma, model_style, need_col, mn_list, y_real):
    dataset = dataset.astype('float64')
    m_epoch = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU加速运算
    train = dataset[:int(len(dataset) * (1 - val_rate))]
    val = dataset[int(len(dataset) * (1 - val_rate)):]
    test = dataset
    Dtr = process(train, batch_size, True, interval)
    Val = process(val, batch_size, True, interval)
    Dte = process(test, batch_size, False, interval)
    if model_style == 0:
        lstm_model = LSTM(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                          batch_size=batch_size, device=device).to(device)
    elif model_style == 1:
        lstm_model = Seq2Seq(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                             batch_size=batch_size, device=device).to(device)
    elif model_style == 2:
        lstm_model = CNN_LSTM(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                              out_channels=32).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    min_val_loss = 1e15

    for epoch in tqdm(range(max_epochs)):
        train_loss = []
        lstm_model.train()
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = lstm_model(seq)
            loss = loss_fn(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # 验证
        lstm_model.eval()
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
            torch.save(lstm_model, f"model-{filename}.pth")  # 保存最好的模型
    print()
    print(f'本次训练损失最小的epoch为{m_epoch},最小损失为{min_val_loss}')
    # 测试
    model = torch.load(f"model-{filename}.pth")
    dataset = y_real
    test_plot(fold_name, mode, title_list, model, t, Dte, filename, dataset, mn_list, device)


def LSTM_main(fold_name, mode, title_list, t, data, max_epochs, interval, val_rate, input_size,
              output_size,
              hidden_size,
              num_layers,
              batch_size, learn_rate, step_size, gamma, model_style, need_col, mn_list, y_real):
    cond = f'{need_col}'
    dataset = data

    ss_time = time.time()  # 开始时间
    train_test(fold_name, mode, title_list, dataset, cond, t, max_epochs, interval, val_rate, input_size,
               output_size,
               hidden_size,
               num_layers,
               batch_size, learn_rate, step_size, gamma, model_style, need_col, mn_list, y_real)
    ed_time = time.time()  # 结束时间
    tt_time = ed_time - ss_time
    print("本次训练总共用时:{}小时:{}分钟:{}秒".format(int(tt_time // 3600), int((tt_time % 3600) // 60),
                                                       int(tt_time % 60)))
