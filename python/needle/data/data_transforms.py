import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION
