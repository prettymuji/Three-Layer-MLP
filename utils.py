import numpy as np
import random
# import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
# from PIL import Image
import pickle
import tarfile
import os
import cupy as cp

def load_CIFAR10():
    # 解压数据集CIFAR10
    if not os.path.exists('cifar-10-batches-py'):
        file_dir = 'cifar-10-python.tar.gz'
        with tarfile.open(file_dir, 'r:gz') as tar:
            tar.extractall()
    
    # 定义加载pickle文件的函数
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # 加载并合并训练数据
    train_data = []
    train_labels = []

    for i in range(1, 5):
        batch_file = f'cifar-10-batches-py/data_batch_{i}'
        batch = unpickle(batch_file)
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    
    # 加载验证集
    val_batch = unpickle('cifar-10-batches-py/data_batch_5')
    val_data = val_batch[b'data']
    val_labels = np.array(val_batch[b'labels'])

    # 加载测试数据
    test_batch = unpickle('cifar-10-batches-py/test_batch')
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    # 归一化
    train_data = train_data.astype('float32') / 255
    val_data = val_data.astype('float32')/255
    test_data = test_data.astype('float32') / 255


    # 现在变量已准备好使用
    print("训练数据形状:", train_data.shape)   
    print("训练标签形状:", train_labels.shape)
    print("验证集数据形状:", val_data.shape)   
    print("验证集标签形状:", val_labels.shape)
    print("测试数据形状:", test_data.shape)    
    print("测试标签形状:", test_labels.shape)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
#--------------------------------------------------------------

#----------------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    test_loss, test_acc = model.large_loss_acc(X_test, y_test)
    return float(test_acc), float(test_loss)

def save_model(model, path):
    params_cpu = {'w1': cp.asnumpy(model.weights1),
                  'b1': cp.asnumpy(model.bias1),
                  'w2': cp.asnumpy(model.weights2),
                  'b2': cp.asnumpy(model.bias2),
                  # 'w3': cp.asnumpy(model.weights3),
                  # 'b3': cp.asnumpy(model.bias3),
                 }
    cp.savez(path, **params_cpu)

def save_info(info, path):
    for k, v in info.items():
        info[k] = np.array([float(i) for i in v])
    cp.savez(path,**info)

def load_model(model, path):
    params = cp.load(path)
    model.weights1 = cp.asarray(params['w1'])
    model.bias1 = cp.asarray(params['b1'])
    model.weights2 = cp.asarray(params['w2'])
    model.bias2 = cp.asarray(params['b2'])
    # model.weights3 = cp.asarray(params['w3'])
    # model.bias3 = cp.asarray(params['b3'])

def get_hyperparams_from_file(path):
    # path = './results/bs=64, h=128, ws=0, lr=0.01, l2_reg=0.001, mode=step'
    info = path.split('/')[-1]
    info = info.split(',')
    bs = int(info[0].split('=')[-1])
    h = int(info[1].split('=')[-1])
    ws = int(info[2].split('=')[-1])
    lr = float(info[3].split('=')[-1])
    l2_reg = float(info[4].split('=')[-1])
    mode = str(info[5].split('=')[-1])
    return bs, h, ws, lr, l2_reg, mode





