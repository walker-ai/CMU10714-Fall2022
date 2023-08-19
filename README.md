# Homework 0

### Question 1

熟悉环境

### Question 2: Loading MNIST data

`parse_mnist(image_filename, label_filename)` 参数1 image.gz，参数2 label.gz

gz文件的格式要参照 [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)，用 `gzip` 工具将 gz 文件以二进制形式读取，得到一个字节对象，然后根据格式进行逐字节解析。

将解析的内容，按照返回格式要求，填充到创建好的 numpy 数组中。
