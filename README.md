## Part 1: Python array operations

- `reshape()`
- `permute()`
- `broadcast_to()`
- `__getitem__()`

在 `ndarray.py` 文件中实现如上四个函数。并且不能重新分配新的内存，而是通过巧妙更改 `stride, shape, offset` 等值来实现如上效果。


这里 `permute` 计算步幅时，与 `compact` 情况下计算的步幅不完全一样，而是步幅也进行相应的 `permute`。

对于 `_getitem_` 函数，这里计算步幅同样不能用 `compact` 计算，按照 `compact` 计算步幅结果是错误的。

对于 `broadcast_to` 函数，实际上只需要找到与 `new_shape` 不同且 `shape[i]=1` 的那一维，对应的 `stride` 设为0就行了。这样的话在这一维上相邻的元素就不会在该维度上移动。

## Part 2: CPU Backend - Compact and setitem

要求实现如下函数：

- `Compact()`
- `EwiseSetitem()`
- `ScalarSetitem()`

即仿照如下的方式来实现通用（无论多少维，即未知数量的for循环）形式：

```c++
cnt = 0;
for (size_t i = 0; i < shape[0]; i++)
    for (size_t j = 0; j < shape[1]; j++)
        for (size_t k = 0; k < shape[2]; k++)
            out[cnt++] = in[strides[0]*i + strides[1]*j + strides[2]*k];
```


`compact` 的作用是将用矩阵形式存储的 `array`` 变为顺序存储形式。即紧凑型存储。

对于Part2，主要难点在于用一个通用实现去代替未知循环数。具体实现在于用一个数组来存储每一个维度下的当前索引数，然后处理当前进位和全局进位。`EwiseSetitem` 的原理和 `compact` 一致。

```c++
/// BEGIN SOLUTION
  size_t dim = shape.size();
  std::vector<size_t> pos(dim, 0);

  for (size_t i = 0; i < a.size; i ++ ) {
    size_t idx = 0;
    for (int j = 0; j < dim; j ++ ) 
      idx += strides[dim - 1 - j] * pos[j];

    out->ptr[idx + offset] = a.ptr[i];
    pos[0] += 1;

    // carry
    for (int j = 0; j < dim; j ++ ) {
      if (pos[j] == shape[dim - 1 - j]) {
        pos[j] = 0;
        if (j != dim - 1) 
          pos[j + 1] += 1;
      }
    }
  }
/// END SOLUTION
```



## Part 3: CPU Backend - Elementwise and scalar operations

实现如下函数：

- `EwiseMul(), ScalarMul()`
- `EwiseDiv(), ScalarDiv()`
- `ScalarPower()`
- `EwiseMaximum(), ScalarMaximum()`
- `EwiseEq(), ScalarEq()`
- `EwiseGe(), ScalarGe()`
- `EwiseLog()`
- `EwiseExp()`
- `EwiseTanh()`


很简单，都是冗余的代码，不详细展开。题目中提到可以用C++的宏或者模版来简化代码。

## Part 4: CPU Backend - Reductions

比较有意思

```c++
// ReduceMax

/// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i ++ ) {
    scalar_t res = a.ptr[i * reduce_size];
    for (size_t j = 0; j < reduce_size; j ++ ) {
      res = std::max(res, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = res;
  }
/// END SOLUTION

// ReduceSum

/// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i ++ ) {
    scalar_t sum = 0;
    
    for (size_t j = 0; j < reduce_size; j ++ ) {
      sum += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum;
    
  }
/// END SOLUTION
```

## Part 5: CPU Backend - Matrix multiplication

- `Matmul()`
- `MatmulTiled()`
- `AlignedDot()`


要求实现一个vanilla版的矩阵乘法和一个tiled版的矩阵乘法，tilde版需要一个额外实现一个辅助函数 `AlignedDot()` 来进行tile和tile之间的矩乘。前两个函数实现没有难度。


分块矩乘中的 `matmul_titled` 有一个点不是很明白，即 `tile` 的内存分布为什么是一块一块的而不是严格按照行存储：

通过后面的的关于convolution implementation的lecture得知，底层存储tile形式的矩阵，为了使其内存是compact的，所以需要一块一块存，块与块之间的内存是相邻的。

```c++
// BEGIN SOLUTION
  float *tmp = new float[TILE * TILE];
  float *A = new float[TILE * TILE];
  float *B = new float[TILE * TILE];

  Fill(out, 0);  // 首先需要将最终结果数组初始化。

  for (size_t i = 0; i < m / TILE; i ++ ) {
    for (size_t j = 0; j < p / TILE; j ++ ) {
      for (size_t l = 0; l < TILE * TILE; l ++ ) 
        tmp[l] = 0;  // 装一个TILE * TILE的结果

      for (size_t k = 0; k < n / TILE; k ++ ) {
        // 指示第(i, k)个tile和第(k, j)个tile做aligndot。
        
        // aligndot(float* A, float* B, float* out) 的结果是将 *A 和 *B做 matmul 然后送入 *out
        // 首先要对*A和*B赋值
        
        // 一维坐标转二维坐标 row = l / TILE, col = l % TILE，但貌似没用到
        for (size_t l = 0; l < TILE * TILE; l ++ ) {
          size_t row = l / TILE, col = l % TILE;
          A[l] = a.ptr[i * TILE * n + k * TILE * TILE + l];  // <-这里的  TILE * TILE 不知道为什么是一块一块存储而不是按行存储
          B[l] = b.ptr[k * TILE * p + j * TILE * TILE + l];
        }
        // 赋值完*A和*B后开始计算
        AlignedDot(A, B, tmp);  
      }
      
      for (size_t l = 0; l < TILE * TILE; l ++ ) {
        out->ptr[i * p * TILE + j * TILE * TILE + l] = tmp[l]; 
      }
    }
// END SOLUTION
```


## Part 6: CUDA Backend - Compact and setitem

实现 CUDA 版的

- `Compact()`
- `EwiseSetitem()`
- `ScalarSetitem()`

✅ [CUDA极简入门指南](https://zhuanlan.zhihu.com/p/34587739) 

CUDA 版的调用一般要写两个函数，一个是在cpu中负责调用gpu中的kernel函数的调用函数，另一个是负责在GPU中并行执行的内核函数，用 `__global__` 声明。

例如本例中的 `Compact()` 实现：

```c++
__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION

  // 这里总的循环次数加起来是gid，但是要分清楚i，j，k分别循环了多少次
  // 因为k在最内层循环，所以按照循环顺序必须倒着累积
  if (gid < size) {
    size_t gid_a = 0;
    size_t tmp = gid;
    for (int i=shape.size-1; i>=0; i--) {
      size_t idx = tmp % shape.data[i];
      tmp /= shape.data[i];
      gid_a += idx * strides.data[i];
    }
    out[gid] = a[gid_a + offset];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}
```

`EwiseSetitem()和ScalarSetitem()` 跟 `Compact()` 类似。


## Part 7: CUDA Backend - Elementwise and scalar operations

同 CPU Backend 一样，实现如下函数，比较简单且冗余，同样可以用 C++ 的宏或者模版简化代码。

- `EwiseMul(), ScalarMul()`
- `EwiseDiv(), ScalarDiv()`
- `ScalarPower()`
- `EwiseMaximum(), ScalarMaximum()`
- `EwiseEq(), ScalarEq()`
- `EwiseGe(), ScalarGe()`
- `EwiseLog()`
- `EwiseExp()`
- `EwiseTanh()`


## Part 8: CUDA Backend - Reductions

原理在于用一个线程完成一个维度（总维度数：`reduce_size`）的缩减。

```c++
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t res = a[gid * reduce_size];
    for (size_t i = 0; i < reduce_size; i ++ ) {
      res = max(res, a[gid * reduce_size + i]);
    }
    out[gid] = res;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t res = 0;
    for (size_t i = 0; i < reduce_size; i ++ ) {
      res += a[gid * reduce_size + i];
    }
    out[gid] = res;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}
```

## Part 9: CUDA Backend - Matrix multiplication

用 CUDA 实现高效矩阵乘法是一个高度热门的话题。这里只给出vanilla版的cuda实现，虽然也可以通过测试样例，但是题目鼓励使用tile等优化方法来进一步优化GEMM，详情可以查看更深入的资料。

```c++
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, size_t M, size_t N, size_t P) {
  // gid是结果矩阵里的紧凑索引，让out矩阵中的每一个元素都用一个线程来完成。
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t res = 0;
    size_t row_a = gid / P, col_b = gid % P;
    for (size_t i = 0; i < N; i ++ ) {
      res += a[row_a * N + i] * b[i * P + col_b];
    }
    out[gid] = res;
  }
}

// 先实现一个vanilla实现版本
void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, size_t M, size_t N,
            size_t P) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,  M,  N, P);
  /// END SOLUTION
}
```

