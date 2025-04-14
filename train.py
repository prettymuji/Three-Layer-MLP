import cupy as cp
from model import model
# warmup+cos模拟退火
def adjust_lr(epoch, warmup_steps, max_lr, min_lr, num_epochs, mode='cos', decay_rate=0.05):
    if epoch <= warmup_steps:
            progress = epoch / warmup_steps
            return max_lr * progress
    else:
        if mode == "step":
            return max(max_lr * (0.1 ** (epoch // (num_epochs//5))), min_lr)
        elif mode == 'cos':
            return max(min_lr + 0.5*(max_lr-min_lr)*(1+cp.cos(cp.pi*epoch/num_epochs)), min_lr)
        else:
            return max(max_lr/(1+epoch*decay_rate), min_lr)

#---------------------------------------------------------------------------------
def train(model, 
          X_train, 
          y_train, 
          X_val, 
          y_val, 
          num_epochs=1000, 
          learning_rate=1e-2,
          target_lr = 3e-4,
          batch_size=64, 
          warmup_steps=10,
          mode = 'cos'
         ):
    
    best_val_loss = float('inf')
    best_params = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr':[]}
    # writer1 = SummaryWriter('log/exp_base/')
    lr = 0.0
    for epoch in range(num_epochs):
        # Shuffle the data
        indices = cp.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            y = cp.eye(10)[y_batch]

            # 前向传播
            model.forward(X_batch)
            # 反向传播并SGD更新权重
            # 调整学习率
            lr = adjust_lr(epoch+1, warmup_steps, learning_rate, target_lr, num_epochs, mode=mode)
            model.backward(X_batch, y, lr)
        
        # 计算损失和精度
        # train_data
        train_loss, train_acc = model.large_loss_acc(X_train, y_train)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        # val_data
        val_loss, val_acc = model.large_loss_acc(X_val, y_val)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        # learning_rate = learning_rate*decay_rate**(epoch//200)
        history['lr'].append(lr)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                'weights1': model.weights1.copy(),
                'bias1': model.bias1.copy(),
                'weights2': model.weights2.copy(),
                'bias2': model.bias2.copy(),
                # 'weights3': model.weights3.copy(),
                # 'bias3': model.bias3.copy()
            }
            
        if epoch%50 == 0:
            print("Epoch {}, Training Loss: {:.4f}, Training Accuracy: {:.4f}, \
                  Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, \
                  lr: {:.4f}".format(epoch+1, train_loss, train_acc, val_loss, val_acc, lr))
        
    
    return best_params, history

if __name__ == "__main__":
    # 载入数据集
    X, y, val_x, val_y, test_X, test_y = utils.load_CIFAR10()
    X_gpu, y_gpu, val_x_gpu, val_y_gpu, test_X_gpu, test_y_gpu = cp.asarray(X), cp.asarray(y), cp.asarray(val_x), cp.asarray(val_y), cp.asarray(test_X), cp.asarray(test_y)
    
    # 定义
    mlp = model.MLP_3layers(
                        input_size = 3072,
                        hidden_size_1 = 256,
                        output_size = 10,
                        activation_function = 'relu', 
                        random_seed = 1999, 
                        L_2_reg = 0.005, 
                        )
    
    # 训练
    best_params, history = train(mlp, 
          X_gpu, 
          y_gpu, 
          val_x_gpu, 
          val_y_gpu, 
          num_epochs=1000, 
          learning_rate=1e-2,
          target_lr = 3e-4,
          batch_size=64, 
          warmup_steps=10,
          mode = 'cos'
         )
    
    # 测试集上评估准确率
    acc,_ = utils.evaluate(mlp, test_X_gpu, test_y_gpu)
    print(f'测试集准确率：{acc: 2f}')
    
    