import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      norm(hidden_dim),
      nn.ReLU(),
      nn.Dropout(drop_prob),  # dropout 应用于每个隐藏层之后，ReLU之后
      nn.Linear(hidden_dim, dim),
      norm(dim),
    )
    
    return nn.Sequential(
      nn.Residual(fn),
      nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    # ### BEGIN YOUR SOLUTION
    modules = [
      nn.Linear(dim, hidden_dim),
      nn.ReLU()
    ]

    for i in range(num_blocks):
      modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    
    modules.append(nn.Linear(hidden_dim, num_classes))

    # 这里一定要按顺序生成并串联各个模块否则会发生错误
    return nn.Sequential(*modules)
    # ### END YOUR SOLUTION



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



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    
    mnist_train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
                                            os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    mnist_test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
                                          os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    # 这里评测case应该有问题，shuffle=False，True 均过不了
    mnist_train_dataloader = ndl.data.DataLoader(mnist_train_dataset, batch_size)
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


if __name__ == "__main__":
    train_mnist(data_dir="../data")
