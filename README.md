## Question 1: Implementing forward computation

直接调用 numpy api 即可.


## Question 2: Implementing backward computation

这里重点拣几个 op 来说明:

1. Matmul:

$$
\begin{equation} 
\begin{split}
  Y &= XW \\
 \text{grad}(X) &= \frac{\partial Y}{\partial X} = dYW^T \\
 \text{grad}(W) &= \frac{\partial Y}{\partial W} = X^TdY \\
\end{split}
\end{equation}
$$

然后检查维度, 用 summation 操作消除多余的维度

```python
a, b = node.inputs
if grad_W.shape != W.shape:
    length = len(grad_W.shape) - len(a.shape)
    grad_W = summation(grad_W, axes = tuple(range(length))
if grad_X.shape  != X.shape:
    ...
```

summation 操作可以起到消除多余维度的作用, 至于为什么要用 summation, 我的理解是根据矩阵微分的性质, summation 可以将其他位置计算的梯度累加到目标位置.

这里可能有疑问, 倘若能够满足矩阵相乘的条件, 那么不就可以反推出 `grad_W.shape` 一定与 `W.shape` 相等吗, 为什么还要做这样的判断. 事实上在 numpy 中, 多维矩阵也是可以进行矩阵相乘的, 但是要满足一些条件:


- 两个矩阵的前 $n-2$ 维必须完全相同，即使有不同的位，也至少应该有一位为 1。例如 $(3,2,4,2)$, $(3,2,2,3)$ 前两维完全一致，或者 $(3, 1, 4, 2)$, $(3, 2, 2, 3)$, 第 2 位不同，但有一位为 1.
- 最后两维必须满足二阶矩阵乘法要求。例如 $(3,2,4,2)$, $(3,2,2,3)$ 的后两维可视为 $(4,2)\times(2,3)$ 满足矩阵乘法。
这样的同维矩阵相乘，所得到的维度是：前 $n-2$ 维不变，后2维进行矩阵乘法。
例如: $(3, 2, 4, 2)\times(1, 2, 2, 3) = (3, 2, 4, 3)$; $(6, 6, 5, 4)\times(4, 3) = (6, 6, 5, 3)$


2. Transpose：

```python
X = np.random.randn(1, 2, 3, 4)
np.transpose(X)  # (4, 3, 2, 1)
```

在 numpy 中, `np.transpose()` 是将所有维度进行反转, 而 needle 中如果不指定反转的轴, 则默认反转最后两维，要使得 `out_grad` 与输出维度相同, 则必须用 `needle` 的 `transpose` 算子将其进行转置. 

3. BroadcastTo / Summation:

首先要了解 numpy 中的广播机制, 两个矩阵 $A$, $B$ 进行运算, 如果 shape 不同, 则需要进行广播让其能够进行操作.

可以广播的前提:

- 两个矩阵各维度大小从后往前对比均一致

例子:

```python
A = np.random.randn(2, 2, 3, 4)
B = np.random.randn(3, 4)
C = A + B  # C.shape = (1, 2, 3, 4)

A = np.random.randn(4)
B = np.random.randn(3, 4)
C = A + B  # C.shape = (3, 4)
```

反例:

```python
A = np.random.randn(2, 2, 3, 4)
B = np.random.randn(3, 3)
C = A + B  # C.shape = (1, 2, 3, 4) 
# ValueError: operands could not be broadcast together with shapes (2,2,3,4) (3,3)  # 最后一维不相等
```

- 两个数组存在一些维度大小不相等时, 有一个数组的该不相等维度大小为1

这是对上面那条规则的补充, 虽然存在多个维大小不一致, 但是只要不相等的那些维有一个数组的该大小是1就可以.


例子:
```python
A = np.random.randn(2, 5, 3, 4)
B = np.random.randn(3, 1)
C = A + B  # C.shape = (2, 5, 3, 4)

A = np.random.randn(2, 5, 3, 4)
B = np.random.randn(2, 1, 1, 4)
C = A + B  # C.shape = (2, 5, 3, 4)

A = np.random.randn(1)
B = np.random.randn(3, 4)
C = A + B  # C.shape = (3, 4)
```

反例:
```python
A = np.random.randn(2, 5, 3, 4)
B = np.random.randn(2, 4, 4, 1)
C = A + B  
# ValueError: operands could not be broadcast together with shapes (2,5,3,4) (2,4,4,1)  # 倒数第二维不相等
```

BroadCastTo 和 Summation 梯度计算的原理是相同的, `out_grad` 的维度与输入不一致, 需要找出具体哪一个轴不一致, 然后用 `broadcast_to` 或 `summation` 将不一致的轴进行扩展或求和去除.
## Question 3: Topological sort

按照题目要求, 以给定的 `node_list` 为拓扑序列终点, 结合辅助函数 `topo_sort_dfs(node, visited, topo_order)` 进行后序深度优先遍历即可. 

