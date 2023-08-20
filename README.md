# Homework 0

## Question 1

熟悉环境

## Question 2: Loading MNIST data

`parse_mnist(image_filename, label_filename)` 参数1 image.gz，参数2 label.gz

gz文件的格式要参照 [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)，用 `gzip` 工具将 gz 文件以二进制形式读取，得到一个字节对象，然后根据格式进行逐字节解析。

将解析的内容，按照返回格式要求，填充到创建好的 numpy 数组中。注意：不要使用循环，效率很低，使用 numpy 提供的工具和内置函数，经测试使用 numpy 比使用循环效率快了近一倍。

## Question 3: Softmax loss

Softmax loss 公式为: $$\ell_{ce}(h(x),y) = -h_y(x) + \log \sum_{j=1}^k \exp (h_j (x))$$

$h$ 是假设函数映射后的结果，大小为 `batch_size * num_classes`，$h_y(x)$ 指的是对于 $x$ 这个样本，它的真实标签 $y$ （列号）所对应的预测值，
$h_j(x)$ 同理。代码中计算 $h_y(x)$ 用到了 numpy 中的花式索引。

`softmax_loss` 要求的返回值是对于一个 batch 的平均损失，因此除以 batch_size 即可。
