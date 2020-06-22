我们已经知道，OneFlow框架的任务函数中采用`blob`作为数据占位符。
OneFlow中可以使用`ConsistentBlob`和`MirroredBlob`两种blob类型。
他们分别对应了不同的策略。

本文将介绍：

* `consistent strategy`的使用及其特点

* `mirrored strategy`的使用及其特点

* `consistent strategy`与`mirrored strategy`的对比及适用场景

## consistent strategy

使用`oneflow.FixedTensorDef`接口定义的blob类型为`ConsistentBlob`，其对应的是`consistent`策略。
在这种策略下，整个分布式系统，无论有多少个节点，对于任务函数而言，逻辑上的输入和输出是唯一的。

在`consistent`策略下，任务函数调用时传递的参数与返回值，与任务函数定义时的blob直接对应即可，如以下例子：
```python
#任务函数的定义
@flow.function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
    #...
    return loss
```
在以上的任务函数定义中，有`images`和`labels`两个`ConsistentBlob`类型的blob占位符，返回loss（也是blob类型）。

那么，我们在调用`train_job`时：
```python
    loss = train_job(images, labels).get()
    if i % 20 == 0: print(loss.ndarray().mean())
```
可见，调用时，任务函数接收的实参和返回值，都与定义时的blob有 **一一对应** 关系，并且通过调用`ndarray`将tensort转变为numpy数组。

## mirrored strategy

我们还可以通过`oneflow.MirroredTensorDef`接口定义`MirroredBlob`类型的占位符，这对应了`mirroed strategy`。

在这种情况下，任务函数定义时的blob，对应调用时的参数，应该是一个`list`，该list中的元素个数，与训练时并行数目决定。

调用时的返回值，也 **不再** 使用`ndarray`接口转变为numpy数据，而是使用`ndarray_list`接口，获取一个保存有多个numpy数组的`list`。

请看下例，我们采用`mirroed strategy`定义任务函数：
```python
@flow.function(get_train_config())
def train_job(images=flow.MirroredTensorDef((50, 1, 28, 28), dtype=flow.float, name="myimages"),
              labels=flow.MirroredTensorDef((50, ), dtype=flow.int32, name="mylabels")):
  logits = lenet(images, train=True)
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  flow.losses.add_loss(loss)
  return loss
```

针对以上的定义，在调用任务函数时：
```python
flow.config.gpu_device_num(2) #使用2个GPU

#images与labels均有100个元素
imgs1 = images[:50]
imgs2 = images[50:]
labels1 = labels[:50]
labels2 = labels1[:50]

loss = train_job([imgs1, imgs2], [labels1, labels2]).get()
loss = loss.ndarray_list()
print(loss[0].mean(), loss[1].mean())
```

## consistent与mirrored的比较及选择

consistent策略在内部使用OneFlow的 **模型并行** 方式驱动，效率上更优，也是OneFlow推荐的默认策略。

mirrored策略内部原理与其它框架（如tensorflow、pytorch）类似，采用 **数据并行** 。

此外，在OneFlow内部，mirrored策略下，支持动态shape：

```python
#定义时的MirroredBlob支持动态shape
@flow.function(func_config)
def ReluJob(a = flow.MirroredTensorDef((5, 2))):
    return ccrelu(a, "my_cc_relu_op")

#调用时，传递的数组shape可以不与定义时的blob完全一致
x1 = np.random.rand(3, 1).astype(np.float32)
x2 = np.random.rand(4, 2).astype(np.float32)
y1, y2 = ReluJob([x1, x2]).get().ndarray_list()
```