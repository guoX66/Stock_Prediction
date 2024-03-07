import re
import numpy as np
from scipy import interpolate
import openpyxl as op
from Muloutput.mutils import LSTM_main
import datetime as dt
import shutil
import os


def train_process(pred_size, file_name, mode, max_epochs, interval, val_rate, hidden_size, num_layers, batch_size,
                  learn_rate, step_size, gamma, is_train, model_style, need_col):
    date_list = []
    title_list = []
    mn_list = []
    wb = op.load_workbook(f'{file_name}')  # 读取excel
    sheet = wb.worksheets[0]  # 读取第一个sheet
    data_list = []
    date1 = list(sheet.values)[1][0][:-2]
    first_date = dt.datetime.strptime(date1, "%Y-%m-%d").date()  # 获取第一个日期
    s_date_list = []
    model_list = ['LSTM', 'Seq2Seq', 'CNN_LSTM']
    fold_name = '多输出' + model_list[model_style] + '-' + mode + '-' + re.findall('database\\\(.*?)\..*?', file_name)[
        0] + f'-预测'
    fold_name = os.path.join(f'./result', f'{fold_name}')
    for row in sheet.iter_rows(min_row=2):  # 从第二行开始读取数据
        data = np.array([i for i in range(len(row) - 1)])
        data = data.astype('float64')
        for j in range(len(row) - 1):
            if j == 0:  # 读取该行第一列的时间
                date2 = str(row[0].value[:-2])
                s_date = dt.datetime.strptime(date2, "%Y-%m-%d").date()
                Days = (s_date - first_date).days
                date_list.append(Days)
                s_date_list.append(s_date)
            if j >= 4:
                data[j] = float(str(row[j + 1].value))
            else:
                data[j] = round(row[j + 1].value, 2)

        data_list.append(data)
    data_list = np.array(data_list)
    date_list = np.array(date_list)
    y_total = [i for i in range(len(data_list[0]))]
    n = date_list[-1]
    t = date_list
    tall = np.arange(0, n + 1)
    tfit = []
    for i in tall:
        date = first_date + dt.timedelta(days=int(i))
        if date.weekday() <= 4:
            tfit.append(i)
    tfit = np.array(tfit)

    for e_num in range(1, len(data_list[0]) + 1):  # 按列插值
        y = []
        title = list(sheet.values)[0][e_num]
        title_list.append(title)
        for i in data_list:
            y.append(i[e_num - 1])

        y = np.array(y).astype('float64')
        tck = interpolate.splrep(t, y, k=3)  # 三次样条插值

        yfit = interpolate.splev(tfit, tck)

        if mode == 'std':  # 标准化
            m1, n1 = np.mean(yfit), np.std(yfit)
            yfit = (yfit - m1) / n1
        elif mode == 'maxmin':  # 最大最小值归一化
            m1, n1 = np.max(yfit), np.min(yfit)
            yfit = (yfit - n1) / (m1 - n1)
        elif mode == 'total':  # 整体归一化
            m1, n1 = np.sum(yfit), np.min(yfit)
            yfit = yfit / m1
        mn_list.append([m1, n1])  # 传入参数用来反归一化
        y_total[e_num - 1] = yfit
    fold_name += f'{title_list[need_col]}'
    try:
        os.mkdir(fold_name)
    except:
        pass
    input_size = len(y_total)  # 输入数据的维度
    output_size = pred_size  # 输出数据的维度

    if is_train:
        y_all = []
        for j in range(len(y_total[0])):
            y_row = []
            for e_num in range(len(y_total)):
                y_row.append(y_total[e_num][j])
            y_all.append(y_row)
        y_all = np.array(y_all)
        title = title_list[need_col]
        LSTM_main(pred_size, fold_name, mode, title, tfit, y_all, max_epochs, interval, val_rate,
                  input_size,
                  output_size,
                  hidden_size,
                  num_layers,
                  batch_size, learn_rate, step_size, gamma, model_style, need_col)
        shutil.move(f"model-{need_col}.pth", f'{fold_name}/model-{need_col}.pth')
    return y_total, title_list, date_list, s_date_list, mode, fold_name, mn_list
