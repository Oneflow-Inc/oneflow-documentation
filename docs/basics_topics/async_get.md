
一般而言，采用异步方式获取训练结果的效率高于同步方式。
本文介绍如何通过调用任务函数的`async_get`方法，异步获取训练结果。

其基本步骤包括：

* 准备回调函数，在回调函数中实现处理任务函数的返回结果的逻辑

* 通过async_get方法注册回调

* OneFlow在合适时机调用注册好的回调，并将任务函数的训练结果传递给该回调

以上工作的前两步由OneFlow用户完成，最后一步由OneFlow框架完成。

## 编写回调函数
回调函数的原型如下：

```python
def cb_func(result):
    #...
```

其中的result，就是任务函数的返回值
比如，对于任务函数：

```python
@flow.function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

返回了一个字典，分别存储了labels和logits两个blob对象。
我们可以实现以下的回调函数，处理两者，计算准确率：

```python
def acc(eval_result):
  global g_total
  global g_correct

  labels = eval_result["labels"]
  logits = eval_result["logits"]

  predictions = np.argmax(logits.ndarray(), 1)
  right_count = np.sum(predictions == labels)
  g_total += labels.shape[0]
  g_correct += right_count
```

再比如，在以下的任务函数中，只是返回了loss。

```python
@flow.function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  #mlp
  #... code not shown
  logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
  
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  
  flow.losses.add_loss(loss)
  return loss
```

对应的回调函数，简单打印平均的loss值：

```python
g_i = 0
def cb_print_loss(result):
  global g_i
  if g_i % 20 == 0:
    print(result.mean())
  g_i+=1
```

## 注册回调函数
调用任务函数，会返回`blob`对象，调用该对象的async_get方法，可以注册我们实现好的回调函数。

```python
train_job(images,labels).async_get(cb_print_loss)
```

OneFlow会在获取到训练结果时，自动调用注册的回调。
