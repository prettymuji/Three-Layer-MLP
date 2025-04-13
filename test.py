import utils, model

if __name__ == "__main__":
    
    path = 'best_params.npz'
    # 载入数据集
    —_, _, _, _, test_X, test_y = utils.load_CIFAR10()
    test_X_gpu, test_y_gpu = cp.asarray(test_X), cp.asarray(test_y)
    
    # 载入模型
    mlp = model.MLP_3layers(
                            input_size = 3072,
                            hidden_size_1 = 256,
                            output_size = 10,
                            activation_function = 'relu', 
                            random_seed = 1999, 
                            L_2_reg = 0.005, 
                            )
    # 导入参数
    utils.load_model(mlp, path)

    
    # 结果评估
    loss, acc = evaluate(mlp, test_X_gpu, test_y_gpu)
    print(f'最优参数组合测试结果: 测试集loss: {loss}, 测试集acc: {acc}')
    