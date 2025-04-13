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
└── results                  # 可视化结果和模型参数
│   ├── params.pth           # 示例参数文件
│   └── figures              # 可视化图片
```
## 模型训练
直接在命令行运行
<pre><code>python train.py</code></pre>
即可得到最优参数组合每50个epoch迭代的loss、acc和learning rate。训练结束之后，将输出模型在测试集上的准确率。
## 测试
直接在命令行运行
<pre><code>python test.py</code></pre>
即可输出最优参数组合的最优模型best_params.npz在测试集上的准确率。
## 参数查找
直接在命令行运行
<pre><code>python hyperparameter_search.py</code></pre>
运行结束后会在该目录下创建新的文件夹‘hyper_params’, 不同超参数对应的收敛信息及相应的模型参数会保存在对应子文件夹下。
## 模型权重下载
