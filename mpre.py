import datetime as dt
import os
import re
import numpy as np
import openpyxl as op
from Muloutput.mtrain import train_process
from Muloutput.mutils import process, write_csv, row2col
import torch
from config import *


def mpredict(need_day, mode, model_style, need_col):
    try:
        os.mkdir('result')
    except:
        pass
    for i in os.walk('database'):
        for file in i[2]:
            file_name = os.path.join('database', file)
            y_total, title_list, date_list, s_date_list, mode, fold_name, mn_list = train_process(need_day,
                                                                                                  file_name,
                                                                                                  mode,
                                                                                                  max_epochs,
                                                                                                  interval,
                                                                                                  val_rate,
                                                                                                  hidden_size,
                                                                                                  num_layers,
                                                                                                  batch_size,
                                                                                                  learn_rate,
                                                                                                  step_size,
                                                                                                  gamma,
                                                                                                  is_train, model_style,
                                                                                                  need_col)

            week_list = ['一', '二', '三', '四', '五', '六', '日']
            wb = op.load_workbook(f'{file_name}')
            sheet = wb.worksheets[0]
            date1 = list(sheet.values)[1][0][:-2]
            today = dt.datetime.now().date()
            future = today + dt.timedelta(days=need_day)
            device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU加速运算
            pred_list = []
            csv_list = [['日期', title_list[need_col]]]

            # model = torch.load(f"model-{e_num}.pth")
            model = torch.load(f'{fold_name}/model-{need_col}.pth')
            model = model.to(device)
            data = row2col(y_total)
            f_input = data[- batch_size - interval + 1:]
            m, n = mn_list[need_col]
            seq = process(mode, f_input, batch_size, False, interval, need_day, pre=True)
            for first_input, _ in seq:
                with torch.no_grad():
                    first_input = first_input.to(device)
                    y_pred = model(first_input)
                    if mode == 'std':
                        y_pred = n * y_pred + m  # 标准化
                    elif mode == 'maxmin':
                        y_pred = (m - n) * y_pred + n  # 最大最小值归一化
                    elif mode == 'total':
                        y_pred = m * y_pred  # 整体归一化

                    y_pred_list = y_pred[-1].tolist()

            date_num = 0
            real_date = 0
            while date_num < need_day:
                date = s_date_list[-1] + dt.timedelta(days=real_date + 1)
                weekday = week_list[date.weekday()]
                real_date += 1
                if weekday == '六' or weekday == '日':
                    continue
                date = str(date) + f',{weekday}'
                day = []
                day.append(round(y_pred_list[date_num], 2))
                csv_list.append([date] + day)
                date_num += 1
            f_name = re.findall('.*?\\\(.*?)-预测', fold_name)[0]
            write_csv(csv_list, f'{fold_name}/{f_name}未来{need_day}天趋势')
            print('*' * 20 + f'文件 {f_name}预测完毕' + '*' * 20)
