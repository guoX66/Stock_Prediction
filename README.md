# 股票短期预测(在更新)

### 对000831中国稀土股票1998-2023年的开盘、最高、最低、收盘数据进行分析、插值、补充、预测

### 预测模型和方法参考子项目：[guoX66/Seq_LSTM (github.com)](https://github.com/guoX66/Seq_LSTM)

### 预测过程图（以开盘为例）：





### 预测误差如下表（LSTM模型）

| output | MSE  | RMSE | MAE  | R2   | MAPE(%) | SMAPE(%) |
|:------ | ---- | ---- | ---- | ---- | ------- | -------- |
| 开盘     | 4.64 | 2.15 | 1.52 | 0.92 | 4.72    | 4.78     |
| 最高     | 6.96 | 2.63 | 1.82 | 0.90 | 5.34    | 5.47     |
| 最低     | 6.35 | 2.52 | 1.83 | 0.89 | 5.65    | 5.79     |
| 收盘     | 7.13 | 2.67 | 1.84 | 0.89 | 5.54    | 5.71     |



# 1、 环境部署

### 用git命令将子项目一并拉取 ,  或将子项目下载后放入本项目中

```bash
git clone --recurse-submodules https://github.com/guoX66/Seq_LSTM.git
```

### 安装 python>=3.10.2，安装依赖

```bash
pip install -r requirements
```

### 然后参考[子项目](https://github.com/guoX66/Seq_LSTM)进行环境部署



# 2、数据处理

### 将数据的各变量因素的取值按行放置，各时间的取值按列放置，保存为excel文件（本项目已将数据放入dataset文件夹中）

### 运行deal.py程序进行数据插值、筛选

```
python deal.py --file datasets/000831.xlsx --output Seq_LSTM/data/static.npz
```

# 3、训练与预测

进入子项目,按照子项目步骤进行

```bash
cd Seq_LSTM
```
