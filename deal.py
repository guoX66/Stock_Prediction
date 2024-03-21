import argparse
import numpy as np
import pandas as pd
import datetime as dt
from scipy import interpolate


def read_and_deal(path):
    # 读取工作簿和工作簿中的工作表
    data_frame = pd.read_excel(path, sheet_name=0)
    va = data_frame.values
    col = data_frame.columns
    first_date = dt.datetime.strptime(str(va[0][0][:-2]), "%Y-%m-%d").date()
    last_date = dt.datetime.strptime(str(va[-1][0][:-2]), "%Y-%m-%d").date()
    t = []
    data = []
    for row in va:
        date2 = str(row[0][:-2])
        s_date = dt.datetime.strptime(date2, "%Y-%m-%d").date()
        Days = (s_date - first_date).days
        t.append(Days)
        row_data = np.array([round(i, 2) for i in row[1:]])
        data.append(row_data)
    t = np.array(t)
    data = np.array(data)
    n = t[-1]
    tall = np.arange(0, n + 1)
    t_fit = []
    for i in tall:
        date = first_date + dt.timedelta(days=int(i))
        if date.weekday() <= 4:
            t_fit.append(i)
    t_fit = np.array(t_fit)
    data_fit = np.zeros((t_fit.shape[0], data.shape[1]))
    for col in range(data.shape[1]):
        col_data = data[:, col]
        tck = interpolate.splrep(t, col_data, k=3)  # 三次样条插值
        y_fit = interpolate.splev(t_fit, tck)
        data_fit[:, col] = y_fit
    return t_fit, data_fit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='datasets/000831.xlsx')
    parser.add_argument('--output', type=str, default='Seq_LSTM/data/static.npz')
    args = parser.parse_args()
    t_fit, data_fit = read_and_deal(args.file)
    np.savez(args.output, t=t_fit, data=data_fit, allow_pickle=True)
