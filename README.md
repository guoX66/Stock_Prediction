# 股票短期预测(在更新)

## 对000831中国稀土股票1998-2023年的开盘、最高、最低、收盘数据进行分析、插值、补充

## 结合多变量多步输入、单变量单步、多部输出，基于多种预测网络构建模型，对股票数据进行短期预测。

## 

# 1、 环境部署

### 首先需安装 python>=3.10.2，然后安装torch ,torchaudio ，torchvision

### 在有nvidia服务的设备上，运行以下命令行命令查看cuda版本

```bash
nvidia-smi
```

### 使用以下命令安装（最后的cu11X要根据cuda版本进行修改）

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```

### 安装后可使用以下命令依次查看torch，cuda版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda)"
```

### 安装其他依赖

```bash
pip install -r requirements
```



# 2、模型训练

## 数据收集

本项目为时间序列一维预测，基于LSTM对excel文件从第二列开始的每一列分别进行预测

将存有需要预测的数据的excel文件放在database文件夹中，并确保一个excel文件只有一个表存储数据。

## 参数设置

可以打开Cfg.yaml，根据注释修改模型参数。默认参数如下：

```yaml
model: LSTM  # LSTM、Seq2Seq、CNN_LSTN
max_epochs: 200  # 迭代数
interval: 30  # 预测间隔
val_rate: 0.8  # 验证集占比
hidden_size: 128  # 隐藏层维度
num_layers: 1  # 循环层数
batch_size: 256 # 数据压缩量
learn_rate: 0.001  # 学习率
step_size: 1  # 学习率递变的步长
gamma: 0.95  # 学习率递增系数，也即每个epoch学习率变为原来的0.95
is_train: True  # 是否训练
```

## 开始训练

设置好参数后，可以直接运行以下命令开始训练,按提示依次输入需要预测的天数、预处理方法、模型选择和预测模式

```bash
python static.py
```

训练结束后，在result文件夹找到对应的训练和预测结果





注意事项：
一、运行过程中，根目录下生成的任何文件请勿改动，否则程序无法顺利运行至结束。
二、若预处理方法、模型选择和预测模式选择了全部，则程序将对各种选择依次进行训练和预测，请耐心等待。
三、Cfg.yaml 中设置的参数将作用于全局。


