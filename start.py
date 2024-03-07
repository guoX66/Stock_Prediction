from mpre import mpredict
from npre import npredict


def predict(need_day, mode, model_style, pre_style, need_col):
    if pre_style == 'all':
        mpredict(need_day, mode, model_style, need_col)  # 多输出预测
        npredict(need_day, mode, model_style, need_col)  # 递归预测
    elif int(pre_style) == 0:
        npredict(need_day, mode, model_style, need_col)
    elif int(pre_style) == 1:
        mpredict(need_day, mode, model_style, need_col)


if __name__ == '__main__':
    need_day = int(input('需要预测的天数为:'))
    mode = input('数据预处理方法(从std、maxmin、total三者中选一个,全部选all):')
    model_style = input('模型选择(LSTM选0、Seq2Seq选1、CNN_LSTN选2、全部选all):')
    pre_style = input('预测模式选择(递归选0、多输出选1、全部选all):')
    if pre_style == 1 or pre_style == 'all':
        need_col = int(input('需要预测的列排序为:')) - 1
    else:
        need_col = 0
    mode_list = ['std', 'maxmin', 'total']
    model_list = [0, 1, 2]
    if mode == 'all' and model_style == 'all':
        for i in mode_list:
            for j in model_list:
                predict(need_day, i, j, pre_style, need_col)
    elif mode == 'all':
        for i in mode_list:
            predict(need_day, i, model_style, pre_style, need_col)

    elif model_style == 'all':
        for j in model_list:
            predict(need_day, mode, j, pre_style, need_col)

    else:
        predict(need_day, mode, int(model_style), pre_style, need_col)
