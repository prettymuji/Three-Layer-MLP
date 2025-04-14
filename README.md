# 基于手写MLP神经网络的CIFAR-10图像分类项目
## 项目简介
本项目不使用PyTorch、TensorFlow等现成的支持自动微分的深度学习框架，仅通过CuPy手工搭建神经网络分类器，实现对CIFAR-10数据集的图像分类任务。
## 数据集
本项目所使用的数据集来自https://www.cs.toronto.edu/~kriz/cifar.html 中的CIFAR-10，在训练开始前，请前往该网页下载cifar-10-python.tar.gz文件于当前目录。
## 仓库结构
```
├── model.py                 # 模型架构
├── train.py                 # 训练框架
├── hyperparameter_search.py # 参数查找
├── test.py                  # 测试最优参数在测试集上的准确率
├── utils.py                 # 数据加载与预处理、模型参数存储与加载、结果可视化
├── best_params.npz          # 最优参数
├── README.md                # 项目说明文件
├── Visualization            # 可视化结果，超参数实验结果表格
```
## 模型训练
直接在命令行运行
<pre><code>python train.py</code></pre>
即可得到最优超参数组合每50个epoch迭代的loss、acc和learning rate。训练结束之后，将输出模型在测试集上的准确率。
## 测试
直接在命令行运行
<pre><code>python test.py</code></pre>
即可输出最优参数组合的最优模型 best_params.npz 在测试集上的准确率。
## 参数查找
1. 直接在命令行运行
<pre><code>python hyperparameter_search.py</code></pre>
运行结束后会在该目录下创建新的文件夹 "hyper_params" , 不同超参数对应的收敛信息及相应的模型参数会保存在对应子文件夹下。
2. Visualization文件夹下的 hyperparameter search.csv 存储了实验所涉及的超参数组合训练结果，可以直接查看。
3. 参数查找环节的所有训练好的模型权重存储在 http://dsw-gateway.cfff.fudan.edu.cn:32080/dsw-15320/lab/tree/zl_dl/hw1/hyper_params 文件夹对应的子文件夹下。
## 模型权重下载
1. 最优模型参数在 best_params.npz 中, 运行`test.py`即可加载该模型并输出在测试集上的准确率。
2. 参数查找环节的所有参数位于http://dsw-gateway.cfff.fudan.edu.cn:32080/dsw-15320/lab/tree/zl_dl/hw1/hyper_params。 可以直接下载整个文件夹于当前目录, 运行`utils.hyper_search()`函数即可依次输出所有超参数组合训练好的模型在测试集上的准确率、最佳准确率对应的超参数组合，相应的结果会储存在'hyperparameter search.csv'文件中。
