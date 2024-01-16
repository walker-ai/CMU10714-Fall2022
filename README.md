## Question 1: Initialization

本问要求实现4个初始化函数. 难度不大, 实现过程中可以调用`init.py`  中的 `rand, randn, constant` 等函数.

## Question 2: NN Library

实现 `nn` 库. 一共有  `Linear, ReLU, Sequential, SofmaxLoss, layerNorm1d, BatchNorm1d, Flatten, Dropout, Residual` 这些 module, 除此之外还需要实现一个算子 `LogSumExp`.

1. `Linear`

这里实现时有一点需要注意 `self.bias` 需要手动 reshape 到对应的形状, 因为 needle 不支持隐式转换.

```python
# __init__
if bias:
    self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).reshape((1, out_features)))
else:
    self.bias = None

# forward
if self.bias:
    return X_weight + self.bias.broadcast_to(X_weight.shape)
else:
    return X_weight
```


2. `ReLU`

直接调用 `ops.relu` 实现即可.

3. `Sequential`

利用父类 `Module` 的 `modules` 属性和 `module()` 方法实现.

4. `LogSumExp`

这里花费了较长时间.

```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=1)
        
        ret = array_api.log(array_api.exp(Z - max_Z).sum(axis=self.axes, keepdims=1)) + max_Z
      
        if self.axes:
          out_shape = [size for i, size in enumerate(Z.shape) if i not in self.axes]
        else:
          out_shape = ()
        
        ret.resize(tuple(out_shape))
        return ret

        '''
        这里要 resize(out_shape)的原因是为了能够和maxz正确的计算，需要keepdims=1
        但是keepdims了以后，计算出来的维度跟正确的维度不符合，经过logsumexp的维度应该是去除了axes的维度
        所以需要resize到正确的维度out_shape
        '''        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION       

        # node就是经过compute计算的结果，即 node = LogSumExp(z)
        Z = node.inputs[0]
        if self.axes:
          s = set(self.axes)
        else:
          s = set(range(len(Z.shape)))
        
        j = 0
        shape_out = [1] * len(Z.shape)
        for i in range(len(Z.shape)):  
          if i not in s:
            shape_out[i] = out_grad.shape[j]
            j += 1

        node_new = node.reshape(shape_out)
        grad_new = out_grad.reshape(shape_out)

        return grad_new * exp(Z - node_new)   
        ### END YOUR SOLUTION
```

其中 `gradient` 的计算, 事实上是不对题目中所描述的 numerically stable 形式的 softmaxloss 进行求导的, 而是对原公式直接求导:

$$
\dfrac{\partial_{out}}{\partial_{z_i}} = \dfrac{e^{z_i}}{e^{\log \sum^m e^{z^k}}} = \dfrac{e^{z_i}}{e^{out}} = e^{z_i - out} = e^{z_i - \log \sum^k e^{z^k}}
$$

其中 $\log \sum^k e^{z^k}$ 就是 `node_new`.

5. `SoftmaxLoss`

用上面实现的 `LogSumExp` 算子实现.

$$
\ell_\text{softmax}(z,y) = \log \sum_{i=1}^k \exp z_i - z_y
$$

6. `LayerNorm1d`

层规范化, paper链接: [Layer Normalization](https://arxiv.org/abs/1607.06450).

$$
y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b
$$

$\textbf{E}[x]$: 经验均值

$\textbf{Var}[x]$: 经验方差(这里使用无偏估计, 因此除以 $N - 1$)


7. `Flatten`

展平层, 直接用 `ops.reshape` 算子实现.

8. `BatchNorm1d`

批量规范化, paper链接: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

对一个小批量进行规范化:

$$
y = w \circ \frac{z_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b
$$

在训练时会计算每个层所有特征的均值和方差的滑动平均值:


$$
\hat{x_{new}} = (1 - m) \hat{x_{old}} + mx_{observed}
$$

然后在测试阶段使用在训练阶段计算的滑动平均值来进行规范化:

$$
y = \frac{(x - \hat{mu})}{((\hat{\sigma}^2_{i+1})_j+\epsilon)^{1/2}}
$$

9. `Dropout`

暂退法, paper链接: [Improving neural networks by preventing co-adaption of feature detectors](https://arxiv.org/abs/1207.0580), 实现简单, 可以用 `init.randb` 生成一个随机矩阵 `mask`, 然后与输入进行逐元素相乘.

10. `Residual`

残差结构, 实现对给定模块 $\mathcal{F}$ 和输入 Tensor $x$ 应用残差或跳跃连接, 返回 $\mathcal{F}(x) + x$.

## Question 3: Optimizer

1. SGD

实现应用了动量法的 SGD:

$$
\begin{split}
    u_{t+1} &= \beta u_t + (1-\beta) \nabla_\theta f(\theta_t) \\
    \theta_{t+1} &= \theta_t - \alpha u_{t+1}
\end{split}
$$

并对权重应用权重衰减:

$$
\theta_{t+1} = (1 - \alpha \lambda)\theta_t - \alpha\nabla_\theta f(\theta_t) 
$$

这里犯了一个错误, 初始化 `u{}` 时, 不能 

```python
for i, param in enumerate(self.params):
    self.u[i] = 0
```

因为在执行一次 `optimizer.step()` 时, 后续的 `batch` 的 `u[i]` 依赖于前面 `batch` 的 `u[i]`. 并且要注意这里的梯度不能原地进行更新, 要新建变量对梯度进行操作.


2. Adam

朴素SGD结合了其他优化方法的 optimizer, paper链接:[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).


$$
\begin{align}
u_{t+1} &= \beta_1 u_t + (1-\beta_1) \nabla_\theta f(\theta_t) \\
v_{t+1} &= \beta_2 v_t + (1-\beta_2) (\nabla_\theta f(\theta_t))^2 \\
\hat{u}_ {t+1} &= u_{t+1} / (1 - \beta_1^t) \quad \text{(bias correction)} \\
\hat{v}_ {t+1} &= v_{t+1} / (1 - \beta_2^t) \quad \text{(bias correction)}\\
\theta_{t+1} &= \theta_t - \alpha \hat{u_{t+1}}/(\hat{v}_{t+1}^{1/2}+\epsilon)
\end{align}
$$

实现过程类似于上面的动量法SGD

## Question 4: Data primitives



### Transforms

首先实现两个数据预处理方法: RandomFlipHorizontal, RandomCrop

1. `RandomFlipHorizontal`

可以应用 `np.flip` 辅助实现.

2. `RandomCrop`

可以用 `np.pad, np.roll` 方法辅助实现, 也可以通过纯计算的方式:

```python
img_pad = np.zeros_like(img)
    H, W = img.shape[0], img.shape[1]
    if abs(shift_x) >= H or abs(shift_y) >= W:
        return img_pad
    img_pad[max(0, -shift_x):min(H - shift_x, H), 
            max(0, -shift_y):min(W - shift_y, W), :] = img[max(0, shift_x):min(H + shift_x, H), 
                                                           max(0, shift_y):min(W + shift_y, W), :]
    return img_pad
```

### Dataset

首先以 MNIST 为例实现数据集类 `MNISTDataset`, 需要调用之前实现的 `parse_mnist` 解析数据集的方法. 另外这里涉及到一个python语法糖, 对于一个类, 如果它能够被索引, 例如 `dataset[index]`, 实际上调用的是这个对象内部实现的 `__getitem__` 方法.


### Dataloader

dataloader提供了一个能够批量迭代数据集对象的接口, 它被要求是一个可迭代的对象, 同时也是一个迭代器. 这就要求它同时要实现 `__next__` 和 `__iter__` 方法. 在 `__next__` 迭代结束后一定要 `raise StopIteration`.

这里实现过程中犯了一个错误, 我没有开辅助变量 `self.start = 0` 来帮助迭代计数, 而是将列表 `self.ordering` 直接 `pop(0)` 的方式进行 `__next__` 的迭代.

## Question 5: Complete MLPResNet

### ResidualBlock

要求实现的残差结构在 `figure` 文件夹下. 按照结构实现即可.

### MLPResNet

同上, 按照结构实现.

### Epoch

单轮训练过程:

```python
def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    
    
    loss_function = nn.SoftmaxLoss()
    correct, loss_sum, num_samples, num_step = 0., 0., 0, 0

    if opt:
      # train
      model.train()
    else:
      model.eval()

    for X, y in dataloader:
      if opt:
        opt.reset_grad()

      pred = model(X)
      correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()
      loss = loss_function(pred, y)
      
      if opt:
        loss.backward()
        opt.step()
      
      loss_sum += loss.numpy()
      num_step += 1
      num_samples += X.shape[0]

    return (1 - correct / num_samples), loss_sum / num_step
    ### END YOUR SOLUTION
```


### Train Mnist

完整训练流程:

```python
def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    
    mnist_train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
                                            os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    mnist_test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
                                          os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    # 这里评测case应该有问题，shuffle=False，True 均过不了
    mnist_train_dataloader = ndl.data.DataLoader(mnist_train_dataset, batch_size, shuffle=True)
    mnist_test_dataloader = ndl.data.DataLoader(mnist_test_dataset, batch_size)

    dim = len(mnist_train_dataset[0][0])

    net = MLPResNet(dim, hidden_dim)

    opt = optimizer(net.parameters(), lr=lr, weight_decay=weight_decay)

    acc_train, loss_train, acc_test, loss_test = 0, 0, 0, 0

    # train
    for _ in range(epochs):
      acc_trian, loss_train = epoch(mnist_train_dataloader, net, opt)
      acc_test, loss_test = epoch(mnist_test_dataloader, net)

    return (acc_train, loss_train, acc_test, loss_test)

    ### END YOUR SOLUTION
```
