import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

import gzip

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        if flip_img:
          return np.flip(img, axis=1)
        else:
          return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        ## Note: np.random.randint(low, high, size) low，下界，包含；high，上界，不包含
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION

        H, W = img.shape[0], img.shape[1]

        padding = self.padding

        pad_width = ((padding, padding), (padding, padding), (0, 0))

        img = np.pad(img, pad_width, mode='constant')

        # 首先将原图按照shift_x, shift_y偏移
        ## 这里注意一个点, np.roll(img, shift(shift_x, shift_y)) shift_x 为正是向下偏移，shift_y为正向右偏移
        ## 与题目中shift_x, shift_y方向相反（题目也没有给定偏移方向的规定，纯属debug得到）
        img = np.roll(img, shift=(-shift_x, -shift_y), axis=(0, 1))        

        # 裁减出原来的中心位置，用切片操作
        
        img = img[padding:padding + H, padding:padding + W, :]
        return img
        ## END YOUR SOLUTION

        


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        indices = np.arange(len(dataset))
        if not self.shuffle:
            self.ordering = np.array_split(indices, range(batch_size, len(dataset), batch_size))
        else:
            # shuffle the ordering
            # 这里必须要shuffle indices，（其实应该shuffle orderring也行，只是评测程序是shuffle了indices，如果shuffle ordering会跟样例有些许不同）
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, range(batch_size, len(dataset), batch_size))
            
    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # 用于迭代小批量
        # self.ordering 是个索引
        self.start = 0

        # 这里不能用 self.ordering = np.apply_split ..., 然后 next() 中 indices = self.ordering.pop(0)
        # 因为ordering是初始化时就定好的，即使shuffle也是只每个epoch shuffle一次，顺序是固定
        # 用self.start 辅助迭代

        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION

        ## 实现__next__()时，必须在结束迭代时 raise StopIteration 
        if self.start == len(self.ordering):
          raise StopIteration

        a = self.start
        self.start += 1

        # indices = self.ordering.pop(0)
        # self.dataset[indices[0]] 是一个元组, (X, y)
      
        # indices 是一个part，类似于[[0, 2], [3, 4], [1, 5], ..., [...]] 中的 [0, 2]
        # 此时batch_size = 2

        # [self.dataset[i][0] for i in indices]
        # ret1 = [self.dataset[0][0], self.dataset[2][0]]
        # ret2 = [self.dataset[0][1], self.dataset[2][1]]
        # (ret1, ret2)
        # 这个元组的长度依据于 self.dataset[0] 的长度

        ret = [Tensor(x) for x in self.dataset[self.ordering[a]]]
        return tuple(ret)

        # X = [self.dataset[i][0] for i in indices]
        # y = [self.dataset[i][1] for i in indices]

        ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        ## 初始化父类的transforms，以便子类能够正确继承父类关于dataset的一些基本操作
        super().__init__(transforms)
        self.image_filename = image_filename
        self.label_filename = label_filename
      
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        ## python语法糖，外部用array[0], array[1]索引实际上调用的是内部的__getitem__方法, 如果要array[idx][0]，则在内部需要判断index是int还是tuple
        X, y = self.images[index], self.labels[index]
        
        # apply_transforms 的输入必须得是 (H, W, C) 的格式（RandomFlipHorizontal, RandomCrop）
        # 所以要先reshape成（H, W, C），应用transforms以后再reshape成原来的格式
        if self.transforms:
          X_in = X.reshape((28, 28, -1))
          X_out = self.apply_transforms(X_in)
          X_ret = X_out.reshape((-1, 28 * 28))
          return X_ret, y
        else:
          return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


def parse_mnist(image_filename, label_filename):
    ### BEGIN YOUR CODE
    gz_image = gzip.open(image_filename, "rb")   
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
    ### END YOUR CODE