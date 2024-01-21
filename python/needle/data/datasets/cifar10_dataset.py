import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        

        # ################# 用于读取每个batch的函数（官网提供）#################
        # def unpickle(file):
        #   with open(file, 'rb') as fo:
        #       dict = pickle.load(fo, encoding='bytes')
        #   return dict
        # ################# 用于读取每个batch的函数（官网提供）#################
        
        
        super().__init__(transforms)  # 初始化父类的transforms，以便子类能够正确继承关于dataset的一些基本操作
        X = []
        y = []
        if train:
          # 返回训练集
            for i in range(1, 6):  # 训练集5个batch, 索引[0, 4]
                with open(os.path.join(base_folder, 'data_batch_%d'%i), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
              # NOTE key: b''
                    X.append(dict[b'data'].astype(np.float32))
                    y.append(dict[b'labels'])
        else:
          # 返回测试集
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                X.append(dict[b'data'].astype(np.float32))
                y.append(dict[b'labels'])

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        X /= 255.0
        self.X = X
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        # NOTE: `self.transforms` need input shape like this.
        if self.transforms:
            X_in = X.reshape((-1, 32, 32, 3))
            X_out = self.apply_transforms(X_in)
            X_ret = X_out.reshape(-1, 3, 32, 32)
            return X_ret, y
        else:
            return np.squeeze(X.reshape((-1, 3, 32, 32))), y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION
