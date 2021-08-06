import oneflow as flow
import numpy as np

# 建立一个简单的张量，并做简单的变换
x =  flow.tensor([1.0,2.0,3.0], requires_grad = True)
print(x)
y = x*x
z = x**3
a = y+z
# 用 MSE 来计算 x 与 y 的差距
loss = flow.nn.MSELoss()
out = loss(x, a)
print(out)

# 反向传播，计算导数
out.backward()
print(x.grad)