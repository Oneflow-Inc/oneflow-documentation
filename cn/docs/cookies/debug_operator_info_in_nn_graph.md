## 如何调试定位nn.Graph中的算子信息

目前`nn.Graph` 可以通过`print` 和 `debug` 两个方法联合起来调试定位算子层面的问题。


例子：

```Python
import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

BATCH_SIZE=64
EPOCH_NUM = 1

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True, drop_last=True
)

model = flowvision.models.mobilenet_v2().to(DEVICE)
model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
model.train()

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

class GraphMobileNetV2(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return loss


graph_mobile_net_v2 = GraphMobileNetV2()

print(graph_mobile_net_v2)
for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        loss = graph_mobile_net_v2(x, y)
        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

```



如果在Graph对象调用后	`print(graph_mobile_net_v2)` ，除了网络的结构信息外，还会打印输入输出的张量信息，以及网络中**算子的签名**信息，有如下类似的效果：

```
(GRAPH:GraphMobileNetV2_0:GraphMobileNetV2): (
  (CONFIG:config:GraphConfig(training=True, ))
  (INPUT:_GraphMobileNetV2_0_input.0.0_2:tensor(..., device='cuda:0', size=(64, 3, 32, 32), dtype=oneflow.float32))
  (INPUT:_GraphMobileNetV2_0_input.0.1_3:tensor(..., device='cuda:0', size=(64,), dtype=oneflow.int64))
  (MODULE:model:MobileNetV2()): (
    (INPUT:_model_input.0.0_2:tensor(..., device='cuda:0', is_lazy='True', size=(64, 3, 32, 32), dtype=oneflow.float32))
    (MODULE:model.features:Sequential()): (
      (INPUT:_model.features_input.0.0_2:tensor(..., device='cuda:0', is_lazy='True', size=(64, 3, 32, 32), dtype=oneflow.float32))
      (MODULE:model.features.0:ConvBNActivation()): (
        (INPUT:_model.features.0_input.0.0_2:tensor(..., device='cuda:0', is_lazy='True', size=(64, 3, 32, 32), dtype=oneflow.float32))
        (MODULE:model.features.0.0:Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)): (
          (INPUT:_model.features.0.0_input.0.0_2:tensor(..., device='cuda:0', is_lazy='True', size=(64, 3, 32, 32), dtype=oneflow.float32))
          (PARAMETER:model.features.0.0.weight:tensor(..., device='cuda:0', size=(32, 3, 3, 3), dtype=oneflow.float32, requires_grad=True)): ()
          (OPERATOR: model.features.0.0.weight() -> (out:sbp=(B), size=(32, 3, 3, 3), dtype=(oneflow.float32)), placement=(oneflow.placement(type="cuda", ranks=[0])))
          (OPERATOR: model.features.0.0-conv2d-0(_GraphMobileNetV2_0_input.0.0_2/out:(sbp=(B), size=(64, 3, 32, 32), dtype=(oneflow.float32)), model.features.0.0.weight/out:(sbp=(B), size=(32, 3, 3, 3), dtype=(oneflow.float32))) -> (model.features.0.0-conv2d-0/out_0:(sbp=(B), size=(64, 32, 16, 16), dtype=(oneflow.float32))), placement=(oneflow.placement(type="cuda", ranks=[0])))
          (OPERATOR: System-AutoGrad-model.features.0.0-conv2d-0-FilterGrad(model.features.0.1-normalization-2_grad/dx_0:(sbp=(B), size=(64, 32, 16, 16), dtype=(oneflow.float32)), _GraphMobileNetV2_0_input.0.0_2/out:(sbp=(B), size=(64, 3, 32, 32), dtype=(oneflow.float32))) -> (System-AutoGrad-model.features.0.0-conv2d-0-FilterGrad/filter_diff_0:(sbp=(B), size=(32, 3, 3, 3), dtype=(oneflow.float32))), placement=(oneflow.placement(type="cuda", ranks=[0])))
          (OPERATOR: model.features.0.0.weight_optimizer(model.features.0.0.weight/out:(sbp=(B), size=(32, 3, 3, 3), dtype=(oneflow.float32)), System-AutoGrad-model.features.0.0-conv2d-0-FilterGrad/filter_diff_0:(sbp=(B), size=(32, 3, 3, 3), dtype=(oneflow.float32)), System-Boxing-Identity-331/out:(sbp=(B), size=(1), dtype=(oneflow.float32))) -> (), placement=(oneflow.placement(type="cuda", ranks=[0])))
          (OUTPUT:_model.features.0.0_output.0.0_2:tensor(..., device='cuda:0', is_lazy='True', size=(64, 32, 16, 16), dtype=oneflow.float32))
        )
        (MODULE:model.features.0.1:BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)): (
          (INPUT:_model.features.0.1_input.0.0_2:tensor(..., device='cuda:0', is_lazy='True', size=(64, 32, 16, 16), dtype=oneflow.float32))
          (PARAMETER:model.features.0.1.weight:tensor(..., device='cuda:0', size=(32,), dtype=oneflow.float32, requires_grad=True)): ()
          (PARAMETER:model.features.0.1.bias:tensor(..., device='cuda:0', size=(32,), dtype=oneflow.float32, requires_grad=True)): ()
          (BUFFER:model.features.0.1.running_mean:tensor(..., device='cuda:0', size=(32,), dtype=oneflow.float32)): ()
          (BUFFER:model.features.0.1.running_var:tensor(..., device='cuda:0', size=(32,), dtype=oneflow.float32)): ()
          (BUFFER:model.features.0.1.num_batches_tracked:tensor(..., device='cuda:0', size=(), dtype=oneflow.int64)): ()
          (OPERATOR: model.features.0.1.num_batches_tracked() -> (out:sbp=(B), size=(), dtype=(oneflow.int64)), placement=(oneflow.placement(type="cuda", ranks=[0])))
          (OPERATOR: model.features.0.1-scalar_add-1(model.features.0.1.num_batches_tracked/out:(sbp=(B), size=(), dtype=(oneflow.int64))) -> (model.features.0.1-scalar_add-1/out_0:(sbp=(B), size=(), dtype=(oneflow.int64))), placement=(oneflow.placement(type="cuda", ranks=[0])))
...
```

如上所示，网络中的算子签名信息都以`(OPERATOR:` 开头，接下来是算子的名字，算子的名字后面的括号里是算子的输入的相关信息，并且带有输入tensor的SBP，size以及type信息。 `->` 后面显示的是算子的输出信息，并且附带有输出tensor的SBP，size，type以及placement信息。



如果我们想查看算子的调用栈，那么可以调用`debug`方法中将 `op_repr_with_py_stack` 参数设为`True`:

`graph_mobile_net_v2.debug(op_repr_with_py_stack=True)` 

打印效果如下：

```
...
(OPERATOR: model.features.0.1-scalar_add-1(model.features.0.1.num_batches_tracked/out:(sbp=(B), size=(), dtype=(oneflow.int64))) -> (model.features.0.1-scalar_add-1/out_0:(sbp=(B), size=(), dtype=(oneflow.int64))), placement=(oneflow.placement(type="cuda", ranks=[0])), location=(Python Stack[-2]: 'forward' at '/home/xiacijie/anaconda3/lib/python3.9/site-packages/flowvision/models/mobilenet_v2.py': line 227; Python Stack[-1]: '_forward_impl' at '/home/xiacijie/anaconda3/lib/python3.9/site-packages/flowvision/models/mobilenet_v2.py': line 219;  ... more))
...
```

我们发现，算子签名多了一个 `location=...` 的信息，后面跟着的便是这个算子的调用栈信息。如果不用`debug`设置调用栈的层数，打印的时候默认显示两层，并且框架代码中的大部分调用栈都会被过滤掉，因为这样可以更加清晰的展示出涉及用户代码的调用栈，更加方便用户进行调试。



对于调用栈超过2层的算子，如果想要调整调用栈的展示层数，可以通过设置`debug`方法中的`max_py_stack`参数调整为想要展示的层数，比如调整成4层：

`graph_mobile_net_v2.debug(op_repr_with_py_stack=True, max_py_stack=4)` 

效果如下：

```
...
(OPERATOR: model.features.0.1-scalar_add-1(model.features.0.1.num_batches_tracked/out:(sbp=(B), size=(), dtype=(oneflow.int64))) -> (model.features.0.1-scalar_add-1/out_0:(sbp=(B), size=(), dtype=(oneflow.int64))), placement=(oneflow.placement(type="cuda", ranks=[0])), location=(Python Stack[-4]: '<module>' at '/home/xiacijie/Project/test/test_doc.py': line 53; Python Stack[-3]: 'build' at '/home/xiacijie/Project/test/test_doc.py': line 38; Python Stack[-2]: 'forward' at '/home/xiacijie/anaconda3/lib/python3.9/site-packages/flowvision/models/mobilenet_v2.py': line 227; Python Stack[-1]: '_forward_impl' at '/home/xiacijie/anaconda3/lib/python3.9/site-packages/flowvision/models/mobilenet_v2.py': line 219; ))
...
```

我们发现，调用栈展示了4层。



如果用户不希望框架相关的调用栈被过滤掉的话，可以将`debug` 中的`only_user_py_stack`参数设置为`False`, 这样的话，框架相关代码的调用栈也会一起展示。

`graph_mobile_net_v2.debug(op_repr_with_py_stack=True, max_py_stack_depth=4, only_user_py_stack=False)` 



打印效果如下：

```
...
(OPERATOR: model.features.0.1-scalar_add-1(model.features.0.1.num_batches_tracked/out:(sbp=(B), size=(), dtype=(oneflow.int64))) -> (model.features.0.1-scalar_add-1/out_0:(sbp=(B), size=(), dtype=(oneflow.int64))), placement=(oneflow.placement(type="cuda", ranks=[0])), location=(Python Stack[-4]: '__call__' at '/home/xiacijie/Project/oneflow/python/oneflow/nn/graph/block.py': line 248; Python Stack[-3]: '__block_forward' at '/home/xiacijie/Project/oneflow/python/oneflow/nn/graph/block.py': line 280; Python Stack[-2]: 'forward' at '/home/xiacijie/Project/oneflow/python/oneflow/nn/modules/batchnorm.py': line 127; Python Stack[-1]: '_add_inplace' at '/home/xiacijie/Project/oneflow/python/oneflow/framework/tensor.py': line 128;  ... more))
...
```

我们发现，之前未被显示的 `oneflow/python/oneflow/nn/graph/block.py`和其他框架代码文件中的调用点也一起被展示出来了。