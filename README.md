## Part 1: ND Backend

- `PowerScalar`
- `EWiseDiv`
- `DivScalar`
- `Transpose`
- `Reshape`
- `BroadcastTo`
- `Summation`
- `MatMul`
- `Negate`
- `Log`
- `Exp`
- `ReLU`
- `LogSumEx`
- `Tanh (new)`
- `Stack (new)`
- `Split (new)`

这里直接把hw2的函数都搬过来就行，需要注意的是之前实现的函数可能借助了 `array_api as numpy` 的实现，因此这里可能会报错，需要把相应的函数修改为最原始的实现。

另外这里 `Tanh` 函数的评测有问题，总是通过不了 backward 的判定。

这里的tanh函数的gradient要用这个方法实现：

```python
class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input_data = node.inputs[0].realize_cached_data()
        return out_grad * (1 - array_api.tanh(input_data) ** 2)
```

具体原因应该是跟 `realize_cached_data()` 含义有关。

```python
# stack
def compute(self, args):
    ### BEGIN YOUR SOLUTION
    shape = args[0].shape  # 获取第一个输入数组的形状
    new_shape = list(shape)  # 构建新的形状，插入长度为 len(args) 的轴
    new_shape.insert(self.axis, len(args))

    # 创建一个新的数组，用于存储堆叠后的结果
    out = array_api.empty(
        new_shape, dtype=args[0].dtype, device=args[0].device)

    # 初始化一个用于构建切片的列表
    slices = []

    # 构建切片列表，用于在指定轴上插入输入数组
    for i in range(len(new_shape)):
        if i != self.axis:
            slices.append(slice(new_shape[i]))  # slice(new_shape[i])代表索引这一维度上的所有内容
        else:
            slices.append(0)

    # 遍历每个输入数组，并在指定位置插入到新数组中
    for i in range(len(args)):
        slices[self.axis] = i
        # NOTE reshape
        out[tuple(slices)] = args[i].reshape((1, ) + shape)

    # 返回堆叠后的结果
    return out
        ### END YOUR SOLUTION

def gradient(self, out_grad, node):
    ### BEGIN YOUR SOLUTION
    return split(out_grad, self.axis)
    ### END YOUR SOLUTION


# split
def compute(self, A):
    ### BEGIN YOUR SOLUTION
    n = A.shape[self.axis] 
    new_shape = list(A.shape)
    new_shape.pop(self.axis)
    slices = [slice(0, s) for s in A.shape]
    splits = []
    for i in range(n):
        slices[self.axis] = slice(i, i+1)
        splits.append(A[tuple(slices)].compact().reshape(new_shape))
    return tuple(splits)
    ### END YOUR SOLUTION

def gradient(self, out_grad, node):
    ### BEGIN YOUR SOLUTION
    return stack(out_grad, self.axis)
    ### END YOUR SOLUTION 
```

## Part 2: CIFAR-10 dataset 

完成 CIFAR-10 dataset 加载器。可以仿照 HW2 的 MNIST数据集加载器。

要求在 `ndarray.py` 中实现

- `flip`
- `pad`

方法。

```python
### filp
def flip(self, axes):
    """
    Flip this ndarray along the specified axes.
    Note: compact() before returning.
    """
    ### BEGIN YOUR SOLUTION
    
    # 如果没有指定轴，就是对所有元素进行反转
    
    # 总的来说，思路在要反转的轴上，找到其底层内存位置，即更改offset位置，并反转步幅
    if axes is None:
        axes = range(len(self.strides))
        
    offset_sum = 0
    new_strides = list(self.strides)
    for axis in axes:
        # NOTE -1!!!
        offset_sum += (self.shape[axis] - 1) * self.strides[axis]
        new_strides[axis] = -self.strides[axis]
        
    ret = NDArray.make(
        shape = self.shape,
        strides=tuple(new_strides),
        device=self.device,
        handle=self._handle,
        offset=offset_sum
    )
    return ret.compact()    
    ### END YOUR SOLUTION


### pad
def pad(self, axes):
    """
    Pad this ndarray by zeros by the specified amount in `axes`,
    which lists for _all_ axes the left and right padding amount, e.g.,
    axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
    """
    ### BEGIN YOUR SOLUTION
    new_shape = list(self.shape)
    for i, ax in enumerate(axes):
        new_shape[i] += ax[0] + ax[1]
    # NOTE not self.make!!!
    ret = NDArray.make(tuple(new_shape), device=self.device)
    ret.fill(0)  # 初始化

    # slices 是原始的self的张量，在新开辟的张量（容器）中的索引
    # ax[0]为左侧填充几行，相当于self在新容器中距离左侧几行
    slices = [slice(ax[0], ax[0] + self.shape[i]) for i, ax in enumerate(axes)]
    ret[tuple(slices)] = self
    return ret
    ### END YOUR SOLUTION
```

在 `ops.py` 中实现如下函数：

- `Flip`
- `Dilate`
- `UnDilate`
- `Conv`

`Flip` 可以直接用 `ndarray.py` 中实现的 `flip` 方法。

```python
class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION
```

`Dilate` 的实现跟 `ndarray.py` 中的 `pad` 方法实现思路很类似。

```python
class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        # NOTE check axes
        for ax in self.axes:
            if ax >= len(a.shape):
                return a
        
        # 原理跟pad类似，但是不能直接调用pad 
        
        # 先扩张大小
        new_shape = list(a.shape)
        for ax in self.axes:
            new_shape[ax] += self.dilation * new_shape[ax]
        
        # 分配内存并初始化
        ret = init.zeros(*new_shape, device=a.device)

        # 确定膨胀前数组在新张量中的索引
        slices = [
            # NOTE +1
            slice(0, new_shape[ax], self.dilation+1) if ax in self.axes
            else slice(0, new_shape[ax], 1)
            for ax in range(len(a.shape))
        ]
        ret.cached_data[tuple(slices)] = a
        return ret.cached_data
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION
```

`UnDilate` 为 `Dilate` 逆运算。

```python
class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        # dilate的逆运算，主要就是索引出其中的部分数据达到缩减的目的
        slices = [
            slice(0, a.shape[ax], self.dilation+1) if ax in self.axes
            else slice(0, a.shape[ax])
            for ax in range(len(a.shape))
        ]
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION
```

## Part 3: Convolutional neural network

im2col 实现原理：

[1](https://cloud.tencent.com/developer/article/1781621)、[2](https://github.com/luweiagi/machine-learning-notes/blob/master/docs/model-deployment/matrix-acceleration-algorithm/im2col/im2col.md)、[3](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Convolution.html#id2)

以及本课程关于卷积实现的基础讲解：

https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb



通过 `as_strided()` 来实现 im2col 的转换，以让其内存是连续排列的。

使用 `np.lib.stride_tricks.as_strided()` 创建一个具有这种形式的矩阵。这个函数可以让你指定新矩阵的形状和跨距，新矩阵是用旧矩阵的相同内存创建的。也就是说，它不会进行任何内存拷贝，因此非常高效。但使用时也要小心，因为它是直接为现有数组创建一个新视图，如果不小心，就可能超出数组的边界。

先通过 `as_stride()` 构造 `(H - K + 1) * (W - K + 1) * K * K` 形状的数组。包含所有卷积窗口的图块，然后将其平铺成一个矩阵。

A（原矩阵,6 * 6) -> 经过 as_stride -> (4, 4, 3, 3) 视图

```python
A = np.arange(36, dtype=np.float32).reshape(6,6)
print(A)

[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10. 11.]
 [12. 13. 14. 15. 16. 17.]
 [18. 19. 20. 21. 22. 23.]
 [24. 25. 26. 27. 28. 29.]
 [30. 31. 32. 33. 34. 35.]]

 W = np.arange(9, dtype=np.float32).reshape(3,3)
print(W)

[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]

 B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))
 print(B)

 [[[[ 0.  1.  2.]
   [ 6.  7.  8.]
   [12. 13. 14.]]

  [[ 1.  2.  3.]
   [ 7.  8.  9.]
   [13. 14. 15.]]

  [[ 2.  3.  4.]
   [ 8.  9. 10.]
   [14. 15. 16.]]

  [[ 3.  4.  5.]
   [ 9. 10. 11.]
   [15. 16. 17.]]]


 [[[ 6.  7.  8.]
   [12. 13. 14.]
   [18. 19. 20.]]

  [[ 7.  8.  9.]
   [13. 14. 15.]
   [19. 20. 21.]]

  [[ 8.  9. 10.]
   [14. 15. 16.]
   [20. 21. 22.]]

  [[ 9. 10. 11.]
   [15. 16. 17.]
   [21. 22. 23.]]]


 [[[12. 13. 14.]
   [18. 19. 20.]
   [24. 25. 26.]]

  [[13. 14. 15.]
   [19. 20. 21.]
   [25. 26. 27.]]

  [[14. 15. 16.]
   [20. 21. 22.]
   [26. 27. 28.]]

  [[15. 16. 17.]
   [21. 22. 23.]
   [27. 28. 29.]]]


 [[[18. 19. 20.]
   [24. 25. 26.]
   [30. 31. 32.]]

  [[19. 20. 21.]
   [25. 26. 27.]
   [31. 32. 33.]]

  [[20. 21. 22.]
   [26. 27. 28.]
   [32. 33. 34.]]

  [[21. 22. 23.]
   [27. 28. 29.]
   [33. 34. 35.]]]]
```

实现 `ops.py` 中的卷积算子。

首先先实现 naive版（即stride=1、paddoing=0）的版本

```python
def compute(self, A, B):
    
    N, H, W, C_in = A.shape
    K, _, _, C_out = B.shape
    Ns, Hs, Ws, Cs = A.strides

    inner_dim = K * K * C_in

    new_shape = (N, H-K+1, W-K+1, K, K, C_in)
    new_stride = (Ns, Hs, Ws, Hs, Ws, Cs)
    lhs = A.as_strided(new_shape, new_stride).reshape((N*(H-K+1)*(W-K+1), inner_dim))
    
    out = lhs @ B.reshape((inner_dim, C_out))

    # 这种方式会增加计算量，不推荐
    # out = out.reshape((N, (H-K+1)//s, (W-K+1)//s, C_out))[:, ::s, ::s, :]
    return out.reshape((N, H-K+1, W-K+1, C_out))
```

当存在填充和跨距时，则需要首先对输入A进行padding：

```python
A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
```

然后对于 new_stride 而言，需要更改为 `(Ns, Hs * s, Ws * s, Hs, Ws, Cs)`
对于 new_shape，需要更改为 `(N, (H-K+1)//s, (W-K+1)//s, K, K, C_in)`

然后构造 im2col 矩阵：

```python
# 这里.compact()在 convolution_implementation.ipynb 中提到过，说实际应用中往往不会真的.compact因为这对于large kernel size来说需要分配更多的内存，很昂贵
im2col = A.as_strided(new_shape, new_stride).reshape((N*((H-K+1)//s)*((W-K+1)//s), inner_dim)).compact() 
```

### Convolution backward

类似于matmul的梯度计算：

```python
X.grad = out_grad @ W.transpose
W.grad = X.transpose @ out_grad
```

conv的梯度计算：

```python
X.grad = ~conv(~out_grad, ~W)
W.grad = ~conv(~X, ~out_grad)
```

其中 `~` 指应用一些特殊操作。

```python
def gradient(self, out_grad, node):
    ### BEGIN YOUR SOLUTION
    
    X, W = node.inputs
    K, _, _, _ = W.shape

    if self.stride > 1:
        out_grad = dilate(out_grad, (1, 2), self.stride - 1)
    W_permute = transpose(flip(W, (0, 1)), (2, 3)) # K * K * C_out * C_in
    # out_grad: # N * (H+2P-K+1) * (W+2P-K+1) * C_out
    X_grad = conv(out_grad, W_permute, padding=K-1-self.padding)

    X_permute = transpose(X, (0, 3)) # C_in * H * W * N
    grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H+2P-K+1) * (W+2P-K+1) * N * C_out
    W_grad = conv(X_permute, grad_permute, padding=self.padding) # C_in * H * W * C_out
    W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2)) # H * W * C_in * C_out

    return X_grad, W_grad
    
    ### END YOUR SOLUTION
```


实现 `nn.conv`：

初始化：

```python
def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
    super().__init__()
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        stride = stride[0]
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride

    ### BEGIN YOUR SOLUTION
    
    # 首先初始化 形状为(k,k,i,o) weight，形状为(o,) bias
    shape = (kernel_size, kernel_size, in_channels, out_channels)

    self.weight = Parameter(init.kaiming_uniform(
        in_channels * kernel_size * kernel_size,
        out_channels * kernel_size * kernel_size,
        shape=shape,
        device=device,
        dtype=dtype,
        requires_grad=True))
    
    if bias:
        self.bias = Parameter(
        init.rand(
            int(self.out_channels),
            low = -1.0 / (in_channels * kernel_size ** 2) ** 0.5,
            high = 1.0 / (in_channels * kernel_size ** 2) ** 0.5,
            device=device,
            dtype=dtype,
            requires_grad=True
        )
        )
    else:
        bias = None
    ### END YOUR SOLUTION
```

前向计算：

```python
def forward(self, x: Tensor) -> Tensor:
    ### BEGIN YOUR SOLUTION
    
    # (N,C_in,H,W) -> (N,H,W,C_in)
    x = x.transpose((1, 2)).transpose((2, 3))

    # 首先计算填充，确保输入和输出维度相同。当 2p = k - 1时，可以保证输入和输出维度相同。
    padding = int((self.kernel_size - 1) / 2)

    # 计算卷积
    ret = ops.conv(x, self.weight, stride=self.stride, padding=padding)

    if self.bias:
        ret += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(ret.shape)
    
    return ret.transpose((2, 3)).transpose((1, 2))
    ### END YOUR SOLUTION
```



开始搭模型的时候，之前写的bug就逐渐暴露出来了。首先是broadcast_to，和batchnorm1d，之前写的有点问题。

以及batchnorm1d，类初始化函数里，`self.weight` 的初始化没有添加 `device=device`


并且之前有一些参数没有用Parameter括起来，导致参数量不正确，没有通过测试。


还有flatten函数，debug大半天，ndarray里面的flatten函数不能直接像numpy中直接
reshape成((X.shape[0], -1)：

```python
class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        length = 1
        for i in X.shape[1:]:
            length *= i
        return X.reshape((X.shape[0], length))
        ### END YOUR SOLUTION
```

以及 train cifar10时，报错的 ndarray.py中 NDArray.reshape 函数，

```python
# 报错提示 new_shape 没有 len
def reshape(self, new_shape):
    ### BEGIN YOUR SOLUTION
    return self.compact().as_strided(new_shape, NDArray.compact_strides(new_shape))
    ### END YOUR SOLUTION
```


并且暴露出relu的gradient错误写法，实际上ndarray后端不支持这种reshape：

```python
def gradient(self, out_grad, node):
    ### BEGIN YOUR SOLUTION
    a = node.inputs[0]
    a = relu(a)
    shape = a.shape  

    a = reshape(a, -1)
    b = a.cached_data
    for i in range(len(b)):
        if b[i] > 0:
        b[i] = 1
    b = Tensor(b)
    b = reshape(b, shape)
    
    return out_grad * b
     ### END YOUR SOLUTION

# 改为：
    a = node.inputs[0].realize_cached_data()
    mask = Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
    return out_grad * mask
```


并且 optim.py 中,SGD和Adam的step函数中，不能使用numpy后端。

```python
# grad_data = ndl.Tensor(param.grad.numpy(), device=param.device, dtype='float32').data \
            #      + param.data * self.weight_decay
grad_data = param.grad.data + param.data * self.weight_decay
```


train cifar10 dataset时，一直抽风。



autograd.py 中，compute_gradient_of_variables 函数写的有问题：

```python
### BEGIN YOUR SOLUTION
for node in reverse_topo_order:
    adj_vi = sum(node_to_output_grads_list[node])
    node.grad = adj_vi

    if node.op is None:
    continue

    for i, grad in enumerate(node.op.gradient_as_tuple(node.grad, node)):
    k = node.inputs[i]
    if k not in node_to_output_grads_list:
        node_to_output_grads_list[k] = []
    node_to_output_grads_list[k].append(grad)
### END YOUR SOLUTION

# 改为

### BEGIN YOUR SOLUTION
for node in reverse_topo_order: 
    node.grad = sum_node_list(node_to_output_grads_list[node])
    
    if node.is_leaf():
        continue
    for i, grad in enumerate(node.op.gradient_as_tuple(node.grad, node)):
        input_ =  node.inputs[i]
        if input_ not in node_to_output_grads_list:
            node_to_output_grads_list[input_] = []
        node_to_output_grads_list[input_].append(grad)
### END YOUR SOLUTION
```

附：[LSTM原理讲解](https://zhuanlan.zhihu.com/p/32085405)，跟RNN原理类似。