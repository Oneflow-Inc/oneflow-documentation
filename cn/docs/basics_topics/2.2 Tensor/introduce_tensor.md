# Tensors 张量



## 什么是Tensor

机器学习中，Tensor是我们离不开的数据形式。神经网络中数据表示，往往是通过称为张量（Tensor）的数据存储库完成的。张量（Tensor）可以视为一个容器，可以容纳N维数据。张量和矩阵有所区别，具体而言，张量是矩阵对N维空间的推广。

 ![An example of a scalar, a vector, a matrix and a tensor](https://hadrienj.github.io/assets/images/2.1/scalar-vector-matrix-tensor.png) 

我们分别对以上数据格式进行说明：

-  **Scalar 标量** 

  标量是单个数字。 标量是0维(0D) 张量。因此，它有0个轴，并且等级为0（张量表示“轴数”）。 

  ```python
import numpy as np
  x = np.array(16)
print(x)
  print('A scalar is of rank %d' %(x.ndim))
  ```
  
  ```
  42
  A scalar is of rank 0
```
  
  
  
-  **Vector 向量** 

   向量是一维 (1D) 张量，我们更常听到将其称为数组。向量由一系列数字组成，有1 个轴，秩为1。 

  ```python
  import numpy as np
  x = np.array([1, 1, 2, 3, 5, 8])
  print(x)
  print('A vector is of rank %d' %(x.ndim))
  ```

  ```
  [1 1 2 3 5 8]
  A vector is of rank 1
  ```



-  **Matrix 矩阵** 

   矩阵是秩为 2 的张量，这意味着它有 2 个轴。矩阵排列为数字网格（行和列），从技术上讲，它是一个二维 (2D) 张量。 

  ```python
  import numpy as np
  x = np.array([[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]])
  print(x)
  print('A matrix is of rank %d' %(x.ndim))
  ```

  ```
  [[1 4 7]
   [2 5 8]
   [3 6 9]]
  A matrix is of rank 2
  ```

  

-  **3D Tensor  张量** 

  从技术上讲，上述所有构造都是有效的张量，但是通俗地说，当我们谈到张量时，我们通常是将矩阵的概念推广到**N≥3维**。因此，为了避免混淆，我们通常只将维度在三维或三维以上称为张量。

  ```python
  import numpy as np
  x = np.array([[[1, 4, 7],
                 [2, 5, 8],
                 [3, 6, 9]],
                [[10, 40, 70],
                 [20, 50, 80],
                 [30, 60, 90]],
                [[100, 400, 700],
                 [200, 500, 800],
                 [300, 600, 900]]])
  print(x)
  print('This tensor is of rank %d' %(x.ndim))
  ```

  ```
  [[[  1   4   7]
    [  2   5   8]
    [  3   6   9]]
   [[ 10  40  70]
    [ 20  50  80]
    [ 30  60  90]]
   [[100 400 700]
    [200 500 800]
    [300 600 900]]]
  This tensor is of rank 3
  ```

  



## 如何得到一个Tensor



### 使用Numpy构造（方法似乎还没更新？）

Tensor可以通过 NumPy arrays进行构造。

```python
import oneflow as flow
import numpy as np

data = [[1, 2],[3, 4]]
np_array = np.array(data)
x_np = flow.from_numpy(np_array)
```



### 使用Tensor中的方法构造





### 使用算子



## Tensor 的常见属性

每个Tensor都有flow.dtype， flow.device，和 flow.layout 三个属性。







## 常见的Tensor操作（算子）

## 常见的Tensor操作（算子）

#### 数值计算类

- add(y)

  ```python
  import numpy as np
  import oneflow.experimental as flow
  
  # element-wise add
  x = flow.Tensor(np.random.randn(2,3))
  y = flow.Tensor(np.random.randn(2,3))
  out1 = flow.add(x, y).numpy()
  print("out1 shape",out1.shape)
  
  
  # scalar add
  x = 5
  y = flow.Tensor(np.random.randn(2,3))
  out2 = flow.add(x, y).numpy()
  print("out2 shape",out2.shape)
  
  
  # broadcast add
  x = flow.Tensor(np.random.randn(1,1))
  y = flow.Tensor(np.random.randn(2,3))
  out3 = flow.add(x, y).numpy()
  print("out3 shape",out3.shape)
  
  ```

  

- abs(y)

  ```python
  import oneflow.experimental as flow
  import numpy as np
  flow.enable_eager_execution()
  
  x = flow.Tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
  x = flow.abs(x)
  print('Tensor x:',X)
  
  ```

  







#### 位置操作类



#### 兼有的

