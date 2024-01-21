"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return (out_grad*self.scalar* lhs **(self.scalar - 1), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        grad_a = out_grad / rhs
        grad_b = -out_grad * lhs / (rhs ** 2)
        return (grad_a, grad_b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.divide(a, self.scalar)
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        # 不能借助numpy实现了

        # if self.axes: return array_api.swapaxes(a, *self.axes)
        # else: return array_api.swapaxes(a, -1, -2)

        order = list(range(len(a.shape)))
        if self.axes is None:
            order[-1] = order[-2]
            order[-2] = len(order) - 1
        else:
            order[self.axes[0]] = self.axes[1]
            order[self.axes[1]] = self.axes[0]
        return a.permute(tuple(order))
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        if self.axes: 
            return transpose(out_grad, self.axes)
        else: 
            return transpose(out_grad)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        return reshape(out_grad, shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape) # (10, ) -> (2, 10)
        axes = []
        shape = [1] * (len(self.shape) - len(shape)) + shape
        for i, s in enumerate(self.shape):
            if i >= len(shape) or s != shape[i]:
                axes.append(i)
        return reshape(summation(out_grad, tuple(axes)), node.inputs[0].shape)
        ### END YOUR SOLUTION

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        else:
            # NOTE self.axes maybe int
            if isinstance(self.axes, int):
                return a.sum(self.axes) 
            # NOTE only support sum in a single dim
            for i, axis in enumerate(sorted(list(self.axes))):
                # NOTE -i because each sum() operation will reduce the dimension number
                a = a.sum(axis-i)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        shape_out = [1] * len(shape)
        # (5, 4, 3, 2) (0, 2) -> (4, 2)
        

        if self.axes is not None:
            if isinstance(self.axes, int):
                s = set([self.axes])
            else:
                s = set(self.axes)
        else:
            s = set(range(len(shape)))
        j = 0
        for i in range(len(shape)):
            if i not in s:
                shape_out[i] = out_grad.shape[j]
                j += 1
        result =  broadcast_to(reshape(out_grad, tuple(shape_out)), shape)
        # print(self.axes, out_grad.shape, shape_out, shape)
        return result
        ### END YOUR SOLUTION

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # a: i * j, b: j * k, out: i * k
        lhs, rhs = node.inputs
        grad_a = matmul(out_grad, transpose(rhs))
        grad_b = matmul(transpose(lhs), out_grad)
        if grad_a.shape != lhs.shape: 
            length = len(grad_a.shape) - len(lhs.shape)
            grad_a = summation(grad_a, axes=tuple(range(length)))
        if grad_b.shape != rhs.shape:
            length = len(grad_b.shape) - len(rhs.shape)
            grad_b = summation(grad_b, axes=tuple(range(length)))
        return grad_a, grad_b 
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION



def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return exp(node.inputs[0]) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0.0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        return out_grad * mask
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_data = node.inputs[0].realize_cached_data()
        return out_grad * (1 - array_api.tanh(input_data)**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    # def compute(self, args):
    #     ### BEGIN YOUR SOLUTION
    #     shape = args[0].shape  # 获取第一个输入数组的形状
    #     new_shape = list(shape)  # 构建新的形状，插入长度为 len(args) 的轴
    #     new_shape.insert(self.axis, len(args))

    #     # 创建一个新的数组，用于存储堆叠后的结果
    #     out = array_api.empty(
    #         new_shape, dtype=args[0].dtype, device=args[0].device)

    #     # 初始化一个用于构建切片的列表
    #     slices = []

    #     # 构建切片列表，用于在指定轴上插入输入数组
    #     for i in range(len(new_shape)):
    #         if i != self.axis:
    #             slices.append(slice(new_shape[i]))  # slice(new_shape[i])代表索引这一维度上的所有内容
    #         else:
    #             slices.append(0)

    #     # 遍历每个输入数组，并在指定位置插入到新数组中
    #     for i in range(len(args)):
    #         slices[self.axis] = i
    #         # NOTE reshape
    #         out[tuple(slices)] = args[i].reshape((1, ) + shape)

    #     # 返回堆叠后的结果
    #     return out
    #     ### END YOUR SOLUTION

    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     return split(out_grad, self.axis)
    #     ### END YOUR SOLUTION

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))

        out = array_api.empty(
            new_shape, dtype=args[0].dtype, device=args[0].device)

        slices = []
        for i in range(len(new_shape)):
            if i != self.axis:
                slices.append(slice(new_shape[i]))
            else:
                slices.append(0)
        for i in range(len(args)):
            slices[self.axis] = i
            # NOTE reshape
            out[tuple(slices)] = args[i].reshape((1, ) + shape)
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    # split是stack的逆向操作，反过来

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


def split(a, axis):
    return Split(axis)(a)


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


def flip(a, axes):
    return Flip(axes)(a)


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
        
        # 原理跟pad类似
        
        # 先扩张大小
        new_shape = list(a.shape)
        for ax in self.axes:
            new_shape[ax] += self.dilation * new_shape[ax]
        
        # 分配内存并初始化
        ret = init.zeros(*new_shape, device=a.device)
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


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


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


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION

        # 首先对A进行padding
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))  # 在H和W轴上进行padding

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        Ns, Hs, Ws, Cs = A.strides

        s = self.stride
        inner_dim = K * K * C_in
        
        new_shape = (N, (H-K+1)//s, (W-K+1)//s, K, K, C_in)
        new_stride = (Ns, Hs * s, Ws * s, Hs, Ws, Cs)

        # 这里.compact()在上一个lecture中提到过，说实际应用中往往不会真的.compact因为这对于large kernel size来说需要
        # 分配更多的内存，很昂贵

      
        # 这里要先compact再reshape，不然会报错：
        # Cannot reshape non-compact array without copying memory
        im2col = A.as_strided(new_shape, new_stride).compact()
        im2col = im2col.reshape((N*((H-K+1)//s)*((W-K+1)//s), inner_dim))
        
        out = im2col @ (B.compact().reshape((inner_dim, C_out)))
        
        # 卷积完之后，为了保证图像大小不变，需要对其进行填充，
        

        # 这种方式会增加计算量，不推荐
        # out = out.reshape((N, (H-K+1)//s, (W-K+1)//s, C_out))[:, ::s, ::s, :]

        return out.compact().reshape((N, (H-K+1)//s, (W-K+1)//s, C_out))
        ### END YOUR SOLUTION        

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

        # C_in * H * W * N @ (H+2P-K+1) * (W+2P-K+1) * N * C_out -> C_in * H * W * C_out，沿着N轴进行累加
        W_grad = conv(X_permute, grad_permute, padding=self.padding) # C_in * H * W * C_out
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2)) # H * W * C_in * C_out

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
