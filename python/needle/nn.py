"""The module.
"""
from typing import List, Callable, Any
from typing_extensions import Required
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION

        # kaiming uniform 初始化A^T(in_fea, out_fea)
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))

        # kaiming uniform 初始化bias(out_fea, 1)
        # bias init
        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).reshape((1, out_features)))
        else:
          self.bias = None
      
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_weight = ops.matmul(X, self.weight)
        if self.bias:
          return X_weight + self.bias.broadcast_to(X_weight.shape)
        else:
          return X_weight
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
       
        ### my solution:
        # shape = list(X.shape)
        # print("X.shape = ", shape)
        # if len(shape) > 2:
        #   ret = 1
        #   for item in shape[1:]:
        #     ret = ret * item
        #   return X.reshape((shape[0], ret))
        # else:
        #   return X

        # 可以用更简单的方法
        return X.reshape((X.shape[0], -1))

        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        
        # input: logits, label: y
        # print(logits)
        # print(y)

        # init.one_hot(n, i), n告知每行的长度，y是label标签（每行真实值）
        
        y_one_hot = init.one_hot(len(logits.cached_data[0]), y)
        
        # print("logitgs = ", logits)
        # print("y = ", y)
        # print("y_one_hot = ", y_one_hot)
        Zy = logits * y_one_hot
        # print("Zy = ", Zy)
      
        ret = ops.logsumexp(logits, axes=(1,)) - ops.summation(Zy, axes=(1,))
        
        average_loss = ops.summation(ret) / len(y.cached_data)
        return average_loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), requires_grad=True)
        self.bias = Parameter(init.zeros(dim), requires_grad=True)
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        # batchnorm1d跟layernorm类似，区别在于对每一列求 平均 和 方差

        batch_size = x.shape[0]
        feature_size = x.shape[1]
  
        x_mean = x.sum((0,)) / batch_size  # 要用元组表示，否则后续计算backward不能对轴进行迭代

        x_minus_mean = x - x_mean.broadcast_to(x.shape)
        x_var = ((x_minus_mean ** 2).sum((0,))) / batch_size

        if self.training:
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.data  # 这里要保证维度相等，因此x_mean, x_var的计算首先不reshape((1, feature_size))
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.data

          x_normed = x_minus_mean / ((x_var + self.eps) ** 0.5).broadcast_to(x.shape)
          return x_normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        else:
          x_normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
          return x_normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        
        # 输入是一个二维张量，第一维batch，第二维features
        self.weight = Parameter(init.ones(dim),requires_grad=True)
        self.bias = Parameter(init.zeros(dim),requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        batch_size = x.shape[0]
        feature_size = x.shape[1]
        
        mean = x.sum(axes=(1, )).reshape((batch_size, 1)) / feature_size
        x_minus_mean = x - mean.broadcast_to(x.shape)
        x_std = ((x_minus_mean ** 2).sum(axes=(1, )).reshape((batch_size, 1)) / feature_size + self.eps) ** 0.5
        normed = x_minus_mean / x_std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          # randb，指定概率p，生成一个张量决定元素有p的概率是true, 1-p概率是false
          mask = init.randb(*x.shape, p=1-self.p) # 1-p的概率要除以1-p
          return mask * x / (1 - self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



