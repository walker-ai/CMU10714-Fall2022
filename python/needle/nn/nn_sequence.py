"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Tanh(Module):
  def forward(self, x: Tensor) -> Tensor:
    ### BEGIN YOUR SOLUTION
    return ops.tanh(x)
    ### END YOUR SOLUTION

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.W_ih = Parameter(
          init.rand(
            input_size, 
            hidden_size, 
            low=-1/hidden_size**0.5,
            high=1/hidden_size**0.5,
            device=device, 
            dtype=dtype, 
            requires_grad=True
          )
        )

        self.W_hh = Parameter(
          init.rand(
            hidden_size, 
            hidden_size, 
            low=-1/hidden_size**0.5,
            high=1/hidden_size**0.5, 
            device=device, 
            dtype=dtype, 
            requires_grad=True
          )
        )

        if bias:
          self.bias_ih = Parameter(
            init.rand(
              hidden_size,
              low=-1/hidden_size**0.5,
              high=1/hidden_size**0.5,
              device=device,
              dtype=dtype,
              requires_grad=True
            )
          )

          self.bias_hh = Parameter(
            init.rand(
              hidden_size,
              low=-1/hidden_size**0.5,
              high=1/hidden_size**0.5,
              device=device,
              dtype=dtype,
              requires_grad=True
            )
          )

        self.nonlinearity = ops.tanh if nonlinearity == 'tanh' else ops.relu
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        out = X @ self.W_ih

        if self.bias:
          out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
          out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        if h is not None:
          out += h @ self.W_hh      
        
        return self.nonlinearity(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers

        rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for i in range(num_layers - 1):
          rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        self.rnn_cells = rnn_cells
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, 0)
        hs = ops.split(h0, 0) if h0 is not None else [None] * self.num_layers
        out = []
        for t, x in enumerate(Xs):
          hiddens = []
          for l, model in enumerate(self.rnn_cells):
            x = model(x, hs[l])
            hiddens.append(x)
          out.append(x)
          hs = hiddens
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)

        return out, hs

        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W_ih = Parameter(
          init.rand(
            input_size,
            4*hidden_size,
            low=-1/hidden_size**0.5,
            high=1/hidden_size**0.5,
            device=device,
            dtype=dtype,
            requires_grad=True
          )
        )

        self.W_hh = Parameter(
          init.rand(
            hidden_size,
            4*hidden_size,
            low=-1/hidden_size**0.5,
            high=1/hidden_size**0.5,
            device=device,
            dtype=dtype,
            requires_grad=True
          )
        )

        if bias:
          self.bias_ih = Parameter(
            init.rand(
              4*hidden_size,
              low=-1/hidden_size**0.5,
              high=1/hidden_size**0.5,
              device=device,
              dtype=dtype,
              requires_grad=True
            )
          )

          self.bias_hh = Parameter(
            init.rand(
              4*hidden_size,
              low=-1/hidden_size**0.5,
              high=1/hidden_size**0.5,
              device=device,
              dtype=dtype,
              requires_grad=True
            )
          )

        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        h0, c0 = (None, None) if h is None else h
        hl = self.hidden_size
        out = X @ self.W_ih

        if h0 is not None:
          out += h0 @ self.W_hh
        if self.bias:
          out += self.bias_ih.reshape((1, 4*hl)).broadcast_to((bs, 4*hl))
          out += self.bias_hh.reshape((1, 4*hl)).broadcast_to((bs, 4*hl))
        
        out_list = ops.split(out, 1)  # 此时out_list中有4*hl个，形状为(bs,1)的张量

        # NOTE out_list is a TensorTuple, cannot slice
        i = ops.stack(tuple([out_list[i] for i in range(0, hl)]), 1)
        f = ops.stack(tuple([out_list[i] for i in range(hl, 2*hl)]), 1)
        g = ops.stack(tuple([out_list[i] for i in range(2*hl, 3*hl)]), 1)
        o = ops.stack(tuple([out_list[i] for i in range(3*hl, 4*hl)]), 1)

        # i,f,g,o形状均为 (bs, hl)的张量
        g = self.tanh(g)
        i, f, o = self.sigmoid(i), self.sigmoid(f), self.sigmoid(o)

        c1 = i * g if c0 is None else f * c0 + i * g
        h1 = o * self.tanh(c1)
        return (h1, c1)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers

        lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(num_layers - 1):
          lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        self.lstm_cells = lstm_cells
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, 0)
        h0, c0 = (None, None) if h is None else h
        hs = [None] * self.num_layers if h0 is None else ops.split(h0, 0)  # 用于存储每个layer的输出h和c，如果初始有了h0就不用创建直接分割就行了
        cs = [None] * self.num_layers if c0 is None else ops.split(c0, 0)
        out = []  # 用于存储每个layer的输出

        for t, x in enumerate(Xs):
          hiddens = []
          cells = []
          for l, model in enumerate(self.lstm_cells):
            x, c_out = model(x, (hs[l], cs[l]))
            hiddens.append(x)
            cells.append(c_out)
          out.append(x)
          hs = hiddens
          cs = cells
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)
        cs = ops.stack(cs, 0)
        return out, (hs, cs)

        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # NOTE: Do not use nn.Linear(), the weight do not need grad!
        self.weight = Parameter(
          init.randn(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype
          )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot_vectors = self.one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        # 这里em指num_embedding，因为one-hot下的embedding维度大小等于词表大小num_embeddings
        seq_len, bs, em = one_hot_vectors.shape
        one_hot_vectors = one_hot_vectors.reshape((seq_len*bs, em))
        out = one_hot_vectors @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION