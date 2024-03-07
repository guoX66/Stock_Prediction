import datetime as dt
import os
import re
import numpy as np
import openpyxl as op
from recursion.ntrain import train_process
from recursion.nutils import process, write_csv, invert_difference, row2col
import torch
from config import *


def npredict(need_day, mode, model_style, need_col):
    try:
        os.mkdir('result')
    except:
        pass
    for i in os.walk('database'):
        for file in i[2]:
            file_name = os.path.join('database', file)
            y_total, title_list, date_list, s_date_list, mode, fold_name, mn_list = train_process(
                file_name, mode, max_epochs, interval, val_rate, hidden_size, num_layers, batch_size, learn_rate,
                step_size, gamma, is_train, model_style, need_col)

            week_list = ['一', '二', '三', '四', '五', '六', '日']
            wb = op.load_workbook(f'{file_name}')
            sheet = wb.worksheets[0]
            date1 = list(sheet.values)[1][0][:-2]
            first_date = dt.datetime.strptime(date1, "%Y-%m-%d").date()
            today = dt.datetime.now().date()
            future = today + dt.timedelta(days=need_day)
            weekday = week_list[future.weekday()]
            device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU加速运算
            csv_list = [['日期'] + title_list]
            times = int(need_day / batch_size) + 1
            model = torch.load(f'{fold_name}/model-{need_col}.pth')
            model = model.to(device)
            data = row2col(y_total)
            y_pred_list = []
            data = data.tolist()
            for time in range(need_day):
                f_input = data[- batch_size - interval + 1:]
                f_input = np.array(f_input)
                seq = process(f_input, batch_size, False, interval, pre=True)
                for first_input, _ in seq:
                    with torch.no_grad():
                        first_input = first_input.to(device)
                        y_pred = model(first_input)
                        # print(y_pred[-1])
                        new_list = y_pred.tolist()
                        new_data = new_list[-1]
                        data.append(new_data)
                        pred_result = []
                        for i in range(len(new_data)):
                            m, n = mn_list[i]
                            if mode == 'std':
                                new_pred = round(n * new_data[i] + m, 2)  # 标准化
                            elif mode == 'maxmin':
                                new_pred = round((m - n) * new_data[i] + n, 2)  # 最大最小值归一化
                            elif mode == 'total':
                                new_pred = round(m * new_data[i], 2)  # 整体归一化
                            pred_result.append(new_pred)
                        y_pred_list.append(pred_result)

            date_num = 0
            real_date = 0
            while date_num < need_day:
                date = s_date_list[-1] + dt.timedelta(days=real_date + 1)
                weekday = week_list[date.weekday()]
                real_date += 1
                if weekday == '六' or weekday == '日':
                    continue
                date = str(date) + f',{weekday}'
                csv_list.append([date] + y_pred_list[date_num])
                date_num += 1
            f_name = re.findall('.*?\\\(.*?)-预测', fold_name)[0]
            write_csv(csv_list, f'{fold_name}/{f_name}未来{need_day}天趋势')
            print('*' * 20 + f'文件 {f_name}预测完毕' + '*' * 20)
