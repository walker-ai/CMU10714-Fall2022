"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
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
          self.in_channels * kernel_size * kernel_size,
          self.out_channels * kernel_size * kernel_size,
          shape=shape,
          device=device,
          dtype=dtype,
          requires_grad=True)
        )
        
        if bias:
          self.bias = Parameter(
            init.rand(
              int(self.out_channels),
              low = -1 / (in_channels * kernel_size ** 2) ** 0.5,
              high = 1 / (in_channels * kernel_size ** 2) ** 0.5,
              device=device,
              dtype=dtype,
              requires_grad=True
            )
          )
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        
        # x: N,C,H,W

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