# Homework 0

## Question 1

熟悉环境

## Question 2: Loading MNIST data

`parse_mnist(image_filename, label_filename)` 参数1 image.gz，参数2 label.gz

gz文件的格式要参照 [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), 用 `gzip` 工具将 gz 文件以二进制形式读取, 得到一个字节对象, 然后根据格式进行逐字节解析.

将解析的内容, 按照返回格式要求, 填充到创建好的 numpy 数组中. 注意: 不要使用循环, 效率很低, 使用 numpy 提供的工具和内置函数, 经测试使用 numpy 比使用循环效率快了近一倍.

## Question 3: Softmax loss

Softmax loss 公式为: $$\ell_{ce}(h(x),y) = -h_y(x) + \log \sum_{j=1}^k \exp (h_j (x))$$

$h$ 是假设函数映射后的结果, 大小为 `batch_size * num_classes`, $h_y(x)$ 指的是对于 $x$ 这个样本, 它的真实标签 $y$ （列号）所对应的预测值,
$h_j(x)$ 同理. 代码中计算 $h_y(x)$ 用到了 numpy 中的花式索引.

`softmax_loss` 要求的返回值是对于一个 batch 的平均损失, 因此除以 batch_size 即可.


## Question 4: Stochastic gradient descent for softmax regression

题目要求实现小批量随机梯度下降:

1. 对于每个 batch $B$, 都有 $X\in \mathbb{R}^{B\times n}, y\in \lbrace 1, \ldots, k \rbrace ^B$, $n$ 代表 `input_dim`，这里是 28 * 28. 以 batch 为步长，对训练集进行随机梯度下降

2. 更新参数  $\theta \in \mathbb{R}^{n\times k}$, ($k$代表类别数, 这里是 10.): $\theta = \theta - \dfrac{\alpha}{\beta}X^{T}(Z - I_y)$

实现公式时, 要理解清楚每个变量代表的含义. $Z = \text{normalize}(\exp (X\theta))$ 等价于 $Z = \text{softmax}(X\theta)$

$\theta$ 代表所需要更新的参数, $X$ 代表输入样本, $I_y\in \mathbb{R}^{B\times k}$ 代表真实标签转化为独热向量后拼接成的矩阵.


## Question 5: SGD for a two-layer neural netwok

与 Question 4 类似, 要求对一个两层的神经网络做小批量梯度下降:

$$
\begin{equation}
\begin{split}
Z_1 \in \mathbb{R}^{m \times d} & = \mathrm{ReLU}(X W_1) \\
G_2 \in \mathbb{R}^{m \times k} & = \text{normalize}(\exp(Z_1 W_2)) - I_y \\
G_1 \in \mathbb{R}^{m \times d} & = \mathrm{1}\lbrace Z_1 > 0\rbrace \circ (G_2 W_2^T)
\end{split}
\end{equation}
$$


令 $\mathrm{1}\lbrace Z_1 > 0\rbrace$ 为一个二进制矩阵, 其每个元素的值取决于 $Z_1$ 中对应位置上的元素是否严格大于零, 且 $\circ$ 表示逐元素乘法. 则目标函数的梯度如下:

$$
\begin{equation}
\begin{split}
\nabla_{W_1} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} X^T G_1  \\
\nabla_{W_2} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_1^T G_2.  \\
\end{split}
\end{equation}
$$

确定每个变量的含义后, 仿照 Question 4 即可.

## Question 6: Softmax regression in C++

将问题4中的python代码翻译成C++版本, 前提不能使用任何库. 在实现矩阵乘法时, 用嵌套循环代替 numpy 中的 dot 操作.
