import re
import numpy as np
from scipy import interpolate
import openpyxl as op
import datetime as dt
import shutil
import os
from recursion.nutils import LSTM_main, difference


def train_process(file_name, mode, max_epochs, interval, val_rate, hidden_size, num_layers, batch_size,
                  learn_rate, step_size, gamma, is_train, model_style, need_col):
    date_list = []
    title_list = []
    mn_list = []
    wb = op.load_workbook(f'{file_name}')
    sheet = wb.worksheets[0]
    data_list = []
    date1 = list(sheet.values)[1][0][:-2]
    first_date = dt.datetime.strptime(date1, "%Y-%m-%d").date()
    s_date_list = []
    model_list = ['LSTM', 'Seq2Seq', 'CNN_LSTM']
    fold_name = '递归' + model_list[model_style] + '-' + mode + '-' + re.findall('database\\\(.*?)\..*?', file_name)[
        0] + '-预测'
    fold_name = os.path.join(f'./result', f'{fold_name}')
    try:
        os.mkdir(fold_name)
    except:
        pass
    for row in sheet.iter_rows(min_row=2):
        data = np.array([i for i in range(len(row) - 1)])
        data = data.astype('float64')
        for j in range(len(row) - 1):
            if j == 0:
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
    y_real = [i for i in range(len(data_list[0]))]
    n = date_list[-1]
    t = date_list
    tall = np.arange(0, n + 1)
    tfit = []
    for i in tall:
        date = first_date + dt.timedelta(days=int(i))
        if date.weekday() <= 4:
            tfit.append(i)
    tfit = np.array(tfit)
    # print(len(tfit))
    for e_num in range(1, len(data_list[0]) + 1):
        y = []
        title = list(sheet.values)[0][e_num]
        title_list.append(title)
        for i in data_list:
            y.append(i[e_num - 1])
        y = np.array(y).astype('float64')
        tck = interpolate.splrep(t, y, k=3)
        yfit = interpolate.splev(tfit, tck)
        y_real[e_num - 1] = yfit
        if mode == 'std':
            m1, n1 = np.mean(yfit), np.std(yfit)
            yfit = (yfit - m1) / n1
        elif mode == 'maxmin':
            m1, n1 = np.max(yfit), np.min(yfit)
            yfit = (yfit - n1) / (m1 - n1)
        elif mode == 'total':
            m1, n1 = np.sum(yfit), np.min(yfit)
            yfit = yfit / m1
        mn_list.append([m1, n1])
        y_total[e_num - 1] = yfit

    try:
        os.mkdir(fold_name)
    except:
        pass
    input_size = len(y_total)  # 输入数据的维度
    output_size = len(y_total)  # 输出数据的维度
    if is_train:
        y_all = []
        for j in range(len(y_total[0])):
            y_row = []
            for e_num in range(len(y_total)):
                y_row.append(y_total[e_num][j])
            y_all.append(y_row)
        y_all = np.array(y_all)
        title = title_list[need_col]
        LSTM_main(fold_name, mode, title_list, tfit, y_all, max_epochs, interval, val_rate, input_size,
                  output_size,
                  hidden_size,
                  num_layers,
                  batch_size, learn_rate, step_size, gamma, model_style, need_col, mn_list, y_real)
        shutil.move(f"model-{need_col}.pth", f'{fold_name}/model-{need_col}.pth')
    return y_total, title_list, date_list, s_date_list, mode, fold_name, mn_list
