# 基于手写MLP神经网络的CIFAR-10图像分类项目
## 项目简介
本项目不使用PyTorch、TensorFlow等现成的支持自动微分的深度学习框架，仅通过CuPy手工搭建神经网络分类器，实现对CIFAR-10数据集的图像分类任务。
## 数据集
来自https://www.cs.toronto.edu/~kriz/cifar.html 中的CIFAR-10。
## 仓库结构
```
├── model.py                 # 模型架构
├── train.py                 # 训练框架
├── hyper_search.py          # 参数查找
├── test.py                  # 测试最优参数在测试集上的准确率
├── utils.py                 # 数据加载与预处理、模型参数存储与加载、结果可视化
├── best_params.npz          # 最优参数
├── cifar-10-python.tar.gz   # 数据集文件
├── README.md                # 项目说明文件
└── results                  # 可视化结果和模型参数
│   ├── params.pth           # 示例参数文件
│   └── figures              # 可视化图片
```
