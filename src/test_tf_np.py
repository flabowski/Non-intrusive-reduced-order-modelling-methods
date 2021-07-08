# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:42:08 2021

@author: florianma
"""
import timeit
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_gpu_available(), tf.test.is_built_with_cuda())


def svd_numpy(X):
    tic = timeit.default_timer()
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    toc = timeit.default_timer()
    print("svd_numpy took {:.4f} s".format(toc-tic))
    return U


def svd_tensorflow(X):
    tic = timeit.default_timer()
    S, U, V = tf.linalg.svd(X, full_matrices=False)
    toc = timeit.default_timer()
    print("svd_tensorflow took {:.4f} s".format(toc-tic))
    return U


def dot_tensorflow(U, X):
    tic = timeit.default_timer()
    X_bar = U @ tf.transpose(U) @ X
    toc = timeit.default_timer()
    print("dot_tensorflow took {:.4f} s".format(toc-tic))
    return X_bar


def dot_numpy1(U, X):
    tic = timeit.default_timer()
    X_bar = U @ U.T @ X
    toc = timeit.default_timer()
    print("dot_numpy1 took {:.4f} s".format(toc-tic))
    return X_bar


def dot_numpy2(U, X):
    tic = timeit.default_timer()
    X_bar = np.dot(np.dot(U, U.T), X)
    toc = timeit.default_timer()
    print("dot_numpy2 took {:.4f} s".format(toc-tic))
    return X_bar


def dot_numpy3(U, X):
    tic = timeit.default_timer()
    X_bar = np.dot(U, np.dot(U.T, X))
    toc = timeit.default_timer()
    print("dot_numpy3 took {:.4f} s".format(toc-tic))
    return X_bar


def np_dot(U, X):
    tic = timeit.default_timer()
    X_bar = np.dot(U.T, X)
    toc = timeit.default_timer()
    print("np.dot took {:.4f} s".format(toc-tic))
    return X_bar


def np_at(U, X):
    tic = timeit.default_timer()
    X_bar = U.T @ X
    toc = timeit.default_timer()
    print("np @ took {:.4f} s".format(toc-tic))
    return X_bar


def np_matmul(U, X):
    tic = timeit.default_timer()
    X_bar = np.matmul(U.T, X)
    toc = timeit.default_timer()
    print("np.matmul took {:.4f} s".format(toc-tic))
    return X_bar


def np_einsum(U, X):
    tic = timeit.default_timer()
    X_bar = np.einsum('ij,jk', U.T, X)
    toc = timeit.default_timer()
    print("np.einsum took {:.4f} s".format(toc-tic))
    return X_bar


def tf_matmul(U, X):
    tic = timeit.default_timer()
    X_bar = tf.matmul(U.T, X)
    toc = timeit.default_timer()
    print("tf.matmul took {:.4f} s".format(toc-tic))
    return X_bar


if __name__ == "__main__":
    print("is_gpu_available?", tf.test.is_gpu_available(),
          tf.test.is_built_with_cuda())
    print(np.__version__, np.__file__)
    print(tf.__version__)

    M = N = 1000
    X = np.random.rand(M, N)
    U = np.random.rand(M, N)
    X_bar = np_dot(U, X)
    X_bar0 = np_at(U, X)
    X_bar1 = np_matmul(U, X)
    X_bar2 = np_einsum(U, X)
    X_bar3 = tf_matmul(U, X)
    print(np.allclose(X_bar, X_bar0))
    print(np.allclose(X_bar, X_bar1))
    print(np.allclose(X_bar, X_bar2))
    print(np.allclose(X_bar, X_bar3))

# False False
# 1.19.2
# 2.2.0
# np.dot took 12.5368 s
# np @ took 14.3303 s
# np.matmul took 14.0634 s
# np.einsum took 0.5937 s
# tf.matmul took 0.0263 s
# True
# True
# True
# True

# Martin
# False False
# 1.19.5
# 1.14.0
# np.dot took 0.0192 s
# np @ took 0.0196 s
# np.matmul took 0.0208 s
# np.einsum took 0.3893 s
# tf.matmul took 0.0280 s
# True
# True
# True
# True

# is_gpu_available? False False
# 1.20.2
# 2.3.0
# np.dot took 0.0658 s
# np @ took 0.0191 s
# np.matmul took 0.0286 s
# np.einsum took 0.4126 s
# tf.matmul took 0.0645 s
# True
# True
# True
# True
# blas_mkl_info:
#     libraries = ['mkl_rt']
#     library_dirs = ['C:/Users/florianma/Anaconda3\\Library\\lib']
#     define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
#     include_dirs = ['C:/Users/florianma/Anaconda3\\Library\\include']
# blas_opt_info:
#     libraries = ['mkl_rt']
#     library_dirs = ['C:/Users/florianma/Anaconda3\\Library\\lib']
#     define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
#     include_dirs = ['C:/Users/florianma/Anaconda3\\Library\\include']
# lapack_mkl_info:
#     libraries = ['mkl_rt']
#     library_dirs = ['C:/Users/florianma/Anaconda3\\Library\\lib']
#     define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
#     include_dirs = ['C:/Users/florianma/Anaconda3\\Library\\include']
# lapack_opt_info:
#     libraries = ['mkl_rt']
#     library_dirs = ['C:/Users/florianma/Anaconda3\\Library\\lib']
#     define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
#     include_dirs = ['C:/Users/florianma/Anaconda3\\Library\\include']
    asd
    for N in [100, 250, 500, 1000, 2000]:
        M = N
        print("SVD of a {:.0f} x {:.0f} Matrix".format(M, N))

        X = np.random.rand(M, N)
        U_np = svd_numpy(X)

        U_tf = svd_tensorflow(X)

        X_bar = dot_tensorflow(U_tf, X)
        print(np.allclose(X, X_bar))
        X_bar = dot_numpy1(U_np, X)
        print(np.allclose(X, X_bar))
        X_bar = dot_numpy2(U_np, X)
        print(np.allclose(X, X_bar))
        X_bar = dot_numpy3(U_np, X)
        print(np.allclose(X, X_bar))

        print()
# SVD of a 2000 x 2000 Matrix
# svd_numpy took 36.8274 s
# svd_tensorflow took 9.1877 s
# dot_tensorflow took 0.6801 s
# dot_numpy1 took 80.0011 s
# dot_numpy2 took 74.0318 s
# dot_numpy3 took 204.5428 s
