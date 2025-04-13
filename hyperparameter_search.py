import cupy as cp
import numpy as np
import os
from train import train
import utils
from model import MLP_3layers

# 参数寻优
def hyperparameter_search(
    X_gpu, 
    y_gpu, 
    val_X_gpu, 
    val_y_gpu,
    batchsize = [64,128],
    hiddensize = [256],
    warmup_steps = [0, 10],
    # learning_rate=[1e-2, 5e-2],
    L2_reg = [0.005, 0.001],
    modes = ['step', 'cos'],
):
    for bs in batchsize:
        for h in hiddensize:
            for ws in warmup_steps:
                for l2_reg in L2_reg:
                    for mode in modes:
                        # 创建文件夹
                        lr=1e-2
                        path = f'./hyper_params/bs={bs}, h={h}, ws={ws}, lr={lr}, l2_reg={l2_reg}, mode={mode}'
                        if os.path.exists(path):
                            print(f'{path}已存在')
                            continue
                        os.makedirs(path, exist_ok=True)
                        # 模型
                        mlp = MLP_3layers(
                        input_size = 3072,
                        hidden_size_1 = h,
                        output_size = 10,
                        activation_function = 'relu', 
                        random_seed = 1999, 
                        L_2_reg = l2_reg, 
                        )
                        params, his = train(
                            mlp, 
                            X_gpu, 
                            y_gpu, 
                            val_X_gpu, 
                            val_y_gpu,
                            warmup_steps = ws,
                            learning_rate=lr,
                            target_lr = 3e-4,
                            batch_size = bs, 
                            num_epochs = 1000, 
                            mode = mode
                            # decay_rate = 0.01
                        )

                        # 存收敛信息
                        his_dir = os.path.join(path, 'history.npz')
                        utils.save_info(his, his_dir)
                        # 存参数
                        params_dir = os.path.join(path, 'params.npz')
                        utils.save_model(mlp, params_dir)
                        print(f'{path}已完成')
                            


                            
if __name__ == "__main__":
    # 载入数据集
    X, y, val_x, val_y, test_X, test_y = utils.load_CIFAR10()
    X_gpu, y_gpu, val_x_gpu, val_y_gpu, test_X_gpu, test_y_gpu = cp.asarray(X), cp.asarray(y), cp.asarray(val_x), cp.asarray(val_y), cp.asarray(test_X), cp.asarray(test_y)
    
    # 超参数查找
    hyperparameter_search(X_gpu, y_gpu, val_x_gpu, val_y_gpu)
    
    