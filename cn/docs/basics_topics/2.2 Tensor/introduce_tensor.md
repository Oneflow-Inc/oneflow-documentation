# Tensors 张量



## 什么是Tensor

​	机器学习中，Tensor是我们离不开的数据形式。神经网络中数据表示，往往是通过称为张量（Tensor）的数据存储库完成的。张量（Tensor）可以视为一个容器，可以容纳N维数据。张量和矩阵有所区别，具体而言，张量是矩阵对N维空间的推广。

​	张量类似于 NumPy 的 ndarray，不同之处在于张量可以在 GPU 或其他硬件加速器上运行。事实上，张量和 NumPy 数组通常可以共享相同的底层内存，从而无需复制数据。除此之外，张量也针对自动微分进行了优化。

​	下面我们将使用Numpy创建一个张量：

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



```python
import oneflow.experimental as flow
import numpy as np

x = flow.Tensor(np.random.rand([5]))
y = flow.ones_like(x)
```



### 使用算子

我们也可以使用算子进行Tensor的构造。

​	

```python
import oneflow.experimental as flow
import numpy as np
flow.enable_eager_execution()

shape = (2,3,)

ones_tensor = flow.ones(shape)
zeros_tensor = flow.zeros(shape)

print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```



## Tensor 的常见属性

 张量属性描述了它们的形状（shape）、数据类型（dtype）和存储它们的设备（device）。 

```python
import oneflow.experimental as flow
import numpy as np
flow.enable_eager_execution()

tensor = flow.Tensor(np.random.randn(2, 6), dtype=flow.float32)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```



## 常见的Tensor操作（算子）

#### 数值计算类

- add(y)

  ```python
  import numpy as np
  import oneflow.experimental as flow
  flow.enable_eager_execution()
  
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

索引和切片

```python
import oneflow.experimental as flow
import numpy as np
flow.enable_eager_execution()

tensor = flow.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

连接多个Tensor.

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```



#### 兼有的

