"""
配置文件
"""
import yaml
with open('Cfg.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)

max_epochs = args['max_epochs']
interval = args['interval']
val_rate = args['val_rate']
hidden_size = args['hidden_size']
num_layers = args['num_layers']
batch_size = args['batch_size']
learn_rate = args['learn_rate']
step_size = args['step_size']
gamma = args['gamma']
is_train = args['is_train']
