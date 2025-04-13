import cupy as cp

# 定义模型
class MLP_3layers:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, activation_function = 'sigmoid', random_seed = 42, L_2_reg = 0, ):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.activation_function = activation_function
        self.random_seed = random_seed
        self.L_2_reg = L_2_reg
        
        # 初始化权重和偏置
        cp.random.seed(self.random_seed)
        self.weights1 = cp.random.randn(self.input_size, self.hidden_size_1)
        self.bias1 = cp.zeros((1, self.hidden_size_1))
        self.weights2 = cp.random.randn(self.hidden_size_1, self.hidden_size_2)
        self.bias2 = cp.zeros((1, self.hidden_size_2))
        self.weights3 = cp.random.randn(self.hidden_size_2, self.output_size)
        self.bias3 = cp.zeros((1, self.output_size))

    def activate(self, x):
        if self.activation_function =='relu':
            return cp.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + cp.exp(-x))
        elif self.activation_function == 'tanh':
            return 2 / (1 + cp.exp(-2*x)) - 1
        else:
            raise ValueError('Invalid activation function')
        
    def softmax(self, o):
        max_o = cp.max(o, axis=1, keepdims=True)
        o = o - max_o
        exp_o = cp.exp(o)
        return exp_o / cp.sum(exp_o, axis=1, keepdims=True)
    
    def activate_derivative(self, x):
        if self.activation_function =='relu':
            return (x > 0).astype('float32')
        elif self.activation_function =='sigmoid':
            x = 1/(1+cp.exp(-x))
            return x*(1-x)
        elif self.activation_function == 'tanh':
            x = 1/(1+cp.exp(-2*x))
            return 4*x*(1-x)
        else:
            raise ValueError('Invalid activation function')


    def forward(self, X):
        # 前向传播
        self.z1 = cp.dot(X, self.weights1) + self.bias1
        self.a1 = self.activate(self.z1)
        self.z2 = cp.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.activate(self.z2)
        self.z3 = cp.dot(self.a2, self.weights3) + self.bias3
        self.output = self.softmax(self.z3)
        return self.output

    def backward(self, X, y, learning_rate): # y: one-hot编码
        bs = X.shape[0]
        # 反向传播求导
        dz3 = (self.output - y)/bs
        dw3 = cp.dot(self.a2.T, dz3)
        db3 = cp.sum(dz3, axis=0, keepdims=True)
        da2 = cp.dot(dz3, self.weights3.T)
        dz2 = da2 * self.activate_derivative(self.z2)
        dw2 = cp.dot(self.a1.T, dz2)
        db2 = cp.sum(dz2, axis=0, keepdims=True)
        da1 = cp.dot(dz2, self.weights2.T)
        dz1 = da1 * self.activate_derivative(self.z1)
        dw1 = cp.dot(X.T, dz1)
        db1 = cp.sum(dz1, axis=0, keepdims=True)
        
        # 更新权重和偏置
        self.weights1 -= learning_rate * (dw1 + self.L_2_reg * self.weights1)
        self.bias1 -= learning_rate * (db1 + self.L_2_reg * self.bias1)
        self.weights2 -= learning_rate * (dw2 + self.L_2_reg * self.weights2)
        self.bias2 -= learning_rate * (db2 + self.L_2_reg * self.bias2)
        self.weights3 -= learning_rate * (dw3 + self.L_2_reg * self.weights3)
        self.bias3 -= learning_rate * (db3 + self.L_2_reg * self.bias3)

    def loss(self, y): # y: 标签
        r = cp.sum(self.weights1**2) + cp.sum(self.weights2**2) + cp.sum(self.weights3**2)
        return -cp.mean(cp.log(self.output[cp.ix_(cp.arange(y.shape[0]), y)])) + self.L_2_reg * r
    
    def accuracy(self, y): # y: 标签
        return cp.mean(cp.argmax(self.output, axis=1) == y)
    
    def large_loss_acc(self, X, y):
        r = cp.sum(self.weights1**2) + cp.sum(self.weights2**2) + cp.sum(self.weights3**2)
        bloss, bacc = [], []
        idx = cp.arange(X.shape[0])
        for i in range(0, X.shape[0], 5000):
            X_batch = X[idx[i:i+5000]]
            y_batch = y[idx[i:i+5000]]
            self.forward(X_batch)
            bloss.append(-cp.mean(cp.log(self.output[cp.ix_(cp.arange(y_batch.shape[0]), y_batch)])))
            bacc.append(self.accuracy(y_batch))
        loss = cp.mean(cp.array(bloss)) + self.L_2_reg * r
        acc = cp.mean(cp.array(bacc))
        return loss, acc

