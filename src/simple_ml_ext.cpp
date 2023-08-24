#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    // 将真实标签转换为独热向量，然后再拼接成矩阵
    float* Iy = new float[m * k];
    for (size_t i = 0; i < m; i ++ ) {
        for (size_t j = 0; j < k; j ++ ) {
            if (y[i] == j) {
                Iy[i * k + j] = 1.0;
            } else {
                Iy[i * k + j] = 0.0;
            }
        }
    }

    for (size_t i = 0; i < m; i += batch) {
        // 创建X_batch和Iy_batch的指针，并进行内存分配
        float* X_batch = new float[batch * n];
        float* Iy_batch = new float[batch * k];

        // 从X和Iy数组中拷贝数据到X_batch和Iy_batch中
        memcpy(X_batch, X + i * n, batch * n * sizeof(float));
        memcpy(Iy_batch, Iy + i * k, batch * k * sizeof(float));

        // X * theta
        float* X_theta = new float[batch * k];
        for (size_t j = 0; j < batch; j ++ ) {
            for (size_t u = 0; u < k; u ++ ) {
                X_theta[j * k + u] = 0;
                for (size_t v = 0; v < n; v ++ ) {
                    X_theta[j * k + u] += X_batch[j * n + v] * theta[v * k + u];
                }
            }
        }

        // normalize(exp(X_theta))
        float* exps = new float[batch * k];
        float* Z = new float[batch * k];
        for (size_t j = 0; j < batch; j ++ ) {
            float sum_exp = 0;
            for (size_t u = 0; u < k; u ++ ) {
                exps[j * k + u] = exp(X_theta[j * k + u]);
                sum_exp += exps[j * k + u];
            }

            for (size_t u = 0; u < k; u ++ ) {
                Z[j * k + u] = exps[j * k + u] / sum_exp;
            }
        }

        // 计算梯度并更新theta
        float* dTheta = new float[n * k];
        std::memset(dTheta, 0, n * k * sizeof(float));

        for (size_t j = 0; j < batch; j ++ ) {
            for (size_t u = 0; u < k; u ++ ) {
                for (size_t v = 0; v < n; v ++ ) {
                    dTheta[v * k + u] += X_batch[j * n + v] * (Z[j * k + u] - Iy_batch[j * k + u]);
                }
            }
        }

        for (size_t j = 0; j < n * k; j ++ ) {
            theta[j] -= lr * dTheta[j] / batch;
        }

        // 释放内存
        delete[] X_batch;
        delete[] Iy_batch;
        delete[] X_theta;
        delete[] exps;
        delete[] Z;
        delete[] dTheta;
    }
    // 释放内存
    delete[] Iy;    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
