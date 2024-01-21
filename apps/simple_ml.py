"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    gz_image = gzip.open(image_filesname, "rb")   
    gz_label = gzip.open(label_filename, "rb")
    
    img_buf = gz_image.read()
    label_buf = gz_label.read()

    # image

    img_magic_number = img_buf[:4].hex()
    num_examples = int(img_buf[4:8].hex(), 16)
    num_rows = int(img_buf[8:12].hex(), 16)
    num_cols = int(img_buf[12:16].hex(), 16)
    input_dim = num_rows * num_cols

    indices_X = 16 + np.arange(num_examples * input_dim)
    img_buf_array = bytearray(img_buf)
    X = np.array([img_buf_array[i] for i in indices_X], dtype=np.float32).reshape(num_examples, input_dim)
    X /= 255
 
    # label
    label_magic_number = label_buf[:4].hex()
    
    indices_y = 8 + np.arange(num_examples)
    label_buf_array = bytearray(label_buf)
    y = np.array([label_buf_array[i] for i in indices_y], dtype=np.uint8)

    return (X, y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size, num_classes = Z.shape[0], Z.shape[1]
  
    Z_y = Z * y_one_hot
    log_sum_exp_Z_i = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1, )))
    loss = log_sum_exp_Z_i - ndl.ops.summation(Z_y, axes=(1, ))
    average_loss = ndl.ops.summation(loss) / batch_size
    
    return average_loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples, input_dim, num_classes, hidden_dim = X.shape[0], X.shape[1], W2.shape[1], W2.shape[0]
    Iy = np.eye(num_classes)[y]

    for i in range(0, num_examples, batch):
      X_batch = X[i:i+batch]
      Iy_batch = Iy[i:i+batch]

      X_batch = ndl.Tensor(X_batch, requires_grad=False)
      Iy_batch = ndl.Tensor(Iy_batch, requires_grad=False)

      Z1 = ndl.ops.relu(ndl.ops.matmul(X_batch, W1))
       
      Z1W2 = ndl.ops.matmul(Z1, W2) # Z1W2 = ReLU(XW1)W2
     
      loss_softmax = softmax_loss(Z1W2, Iy_batch) # loss_softmax(ReLU(XW1)W2, y) 

      loss_softmax.backward()

      W1.data -= lr * ndl.Tensor(W1.grad.numpy().astype(np.float32))
      W2.data -= lr * ndl.Tensor(W2.grad.numpy().astype(np.float32))

    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        # train
        model.train()
    else:
        model.eval()

    for X, y in dataloader:
        if opt:
            opt.reset_grad()

        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.numpy()
        n_step += 1
        n_samplers += X.shape[0]

    return correct / n_samplers, loss_sum / n_step
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        train_acc, train_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=opt)
      
    return train_acc, train_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(dataloader, model, loss_fn())
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    correct, loss_sum, n_step, n_samplers = 0., 0., 0., 0
    if opt:
      model.train()
    else:
      model.eval()

    h = None
    for i in range(0, data.shape[0]-1, seq_len):
      X, y = ndl.data.get_batch(data, i, seq_len, device, dtype)
      if opt:
        opt.reset_grad()
        # NOTE: use
      pred, h = model(X, h)

      if isinstance(h, tuple):
        h = (h[0].detach(), h[1].detach())
      else:
        h = h.detach()

      loss = loss_fn(pred, y)
      correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()
      if opt:
        loss.backward()
        opt.step()
      # NOTE multiply seq_len
      loss_sum += loss.numpy() * y.shape[0]
      n_step += 1
      n_samplers += y.shape[0]

    return correct / n_samplers, loss_sum / n_samplers
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
      train_acc, train_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt, clip=clip, device=device, dtype=dtype)
      print("train_acc = ", train_acc, "train_loss = ", train_loss)
    return train_acc, train_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), device=device, dtype=dtype)
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
