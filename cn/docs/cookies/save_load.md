# OneFlow 的大模型分片保存和加载


## 大规模模型分片存储简介

在模型比较小的时候，比如 100G 以下，还有可能采用单机存储。当模型参数量比较大的时候，这个时候要求的样本数也更大，训练后做 dump 出来的模型也会很大，单机肯定是放不下的。比如，由 DeepSpeed 和 Megatron 驱动的 Megatron 图灵自然语言生成模型（MT-NLG）具有 5300 亿个参数，是迄今为止训练过的最大和最强大的单片 Transformer 语言模型，支持这样的大规模语言模型需要分片保存和加载，不会使用单机内存。此外，在其他 CV、搜索、推荐和广告类等场景下，读取样本量增多和模型复杂度增加都会带来模型存储上的难题。

本文将介绍 OneFlow 的大模型分片保存、加载策略以及使用方法。

## OneFlow 模型分片保存和加载

OneFlow 的大模型分片保存和加载的实现基于全局视角（[Global View](https://docs.oneflow.org/master/cookies/global_tensor.html)）的概念，既利用 Placement 与 SBP 完成模型文件（下文都用 state dict 表示）在各个物理设备上的切分，适用于当模型大到无法在单个设备的内存或显存上容纳下的场景。

### flow.utils.global_view.to_global() 接口介绍

为了更好理解下文保存模型和加载模型两个部分的内容，首先对  `flow.utils.global_view.to_global()`  接口和其实现思路进行分析。区别于现有的 [Tensor.to_global()](https://oneflow.readthedocs.io/en/master/generated/oneflow.Tensor.to_global.html?highlight=to_global%28%29) 模式（可以处理普通的 Tensor），提供了多种类型的输入支持，包括 None、Tensor、List、Tuple、nn.Module 的 state dict 、nn.Graph 的 state dict 和几种类型的任意组合，既将 List/Tuple/Dict 中的输入 Tensor 转换为 Global Tensor。值得注意的是，其传入参数中的 SBP 支持用户自定义一个 `(x, tensor) -> sbp` 的函数来解决不同 Tensor 对应不同 SBP 的需求。

并且，与 to_global() 对应的还有 `flow.utils.global_view.to_local()` 接口。可以参考 API 文档中关于 to_global() 和 to_local() 更[详细的介绍](https://oneflow.readthedocs.io/en/master/utils.global_view.html)。在 `flow.utils.global_view.to_global()` 的[实现](https://github.com/Oneflow-Inc/oneflow/blob/master/python/oneflow/utils/global_view/to_global.py)中，支持了多种输入类型适用于现有的 `Tensor.to_global()` 接口。实现的整体思路大致为检查输入、广播（空）结构，遍历节点、调用回调函数和返回 to_global() 后的结果。

再回到我们关注的地方，这个接口如何做到模型分片保存和加载？比如对于模型并行/流水并行，模型的参数分散在多个 Rank 上，在保存模型前通过  `flow.utils.global_view.to_global()`  将 state dict 里的每个 Tensor 在指定 Placement 上转为 Global Tensor，SBP 的类型为  `flow.sbp.split`，可以设置在特定维度上的切分。同样的，模型也可以按 Split 被加载。当然，SBP 也可以为 Broadcast，支持不同的 SBP 和 Placement 组合。这样，超大规模模型分片存储的问题就被非常好的解决了。

### 保存模型

大致了解  `flow.utils.global_view.to_global()`  接口后，在这一部分演示了如何分片保存模型，代码如下：

```python
# 自定义 get_sbp 函数。
def get_sbp(state_dict, tensor):
    if tensor is state_dict["System-Train-TrainStep"]:
        return BROADCAST
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.weight"]:
        return flow.sbp.split(1)
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.bias"]:
        return BROADCAST
    return flow.sbp.split(0)

model_file_state_dict = flow.utils.global_view.to_global(
    state_dict, placement=model_file_placement, sbp=get_sbp, 
    ) # 使用 sbp=get_sbp 处理特殊的键，也支持指定普通的 SBP。
if flow.env.get_rank() in model_file_placement.ranks:
    flow.save(
        flow.utils.global_view.to_local(model_file_state_dict),
        state_dict_dir,
    ) # state_dict_dir = "./graph_save_load_global_" + str(rank_id)
```

首先，将原模型（state_dict）转化到模型文件的 Placement 和 SBP 上，model_file_placement 为要分片保存模型的设备阵列，也就是将 state dict 按 split(0) 分片到 model_file_placement 上。这里之所以自定义 get_sbp 函数，是因为用户可以传进来一个 `(x, tensor) -> sbp` 的函数来解决特殊 Tensor 对应不同 SBP 的需求。举个例子(当前例子基于 Graph 模式)，对于  `state_dict["System-Train-TrainStep"]`  这种 shape 为 [1] 的 Tensor，我们就不能按 split(0) 分片了，SBP 可以选用 broadcast。而 `state_dict["module_pipeline"]["m_stage3.linear.weight"]`  只能在第 1 维度切分，对于 `state_dict["module_pipeline"]["m_stage3.linear.bias"]` 这种不可切分的小 Tensor(s)，SBP 可以选用 broadcast。这样支持用户 DIY SBP 的处理，更加灵活。

在后面的处理中，使用 `flow.utils.global_view.to_local()` 接口得到 model_file_state_dict  的本地分量，并调用 [save()](https://oneflow.readthedocs.io/en/master/generated/oneflow.save.html?highlight=save) 保存模型。其中，state_dict_dir 是带有设备 id 的目录，需要区分不同设备，推荐一个 rank 对应一个路径，路径名用 rank id 的方式。

### 加载模型

在指定设备上分片保存模型后，加载模型的代码如下：

```python
if cur_rank in model_file_placement.ranks:
    local_state_dict = flow.load(state_dict_dir)
else:
    local_state_dict = None

global_state_dict = flow.utils.global_view.to_global(
     local_state_dict, placement=model_file_placement, sbp=get_sbp,
)
graph_model.load_state_dict(global_state_dict)
```

首先，用 [load()](https://oneflow.readthedocs.io/en/master/generated/oneflow.load.html?highlight=load) 方法在每个保存切片的设备上加载 state dict。对应的，需要把 local rank 上的 state dict 转换到模型文件的 placement 和 sbp 上，得到了 global_state_dict。这一步和保存模型应该是对应的，SBP 和 Placement 也是一致的。最后，global_state_dict 可以成功加载到 graph_model（nn.Graph） 中。当然，nn.Module 和 nn.Graph 处理方法是一致的。

### 将 state dict 加载到 nn.Module 中

除了以上两个特征外，在将 state dict 加载到 nn.Module 时，OneFlow 提供了 SBP 和 Placement 的自动转换。在下面的例子中，首先构造一个 m（nn.Module）对象，然后将 global_state_dict 的 SBP 设置为 split(0)，而 nn.Module 的 SBP 为 broadcast，同时 placement 也放生了变化，OneFlow 会自动做这个转换过程。

```python
import oneflow as flow

m = flow.nn.Linear(2,6)
model_file_placement = flow.placement("cpu", ranks=[0, 1])

state_dict = {"weight":flow.ones(3,2), "bias":flow.zeros(3)}
global_state_dict = flow.utils.global_view.to_global(
    state_dict, placement=model_file_placement, sbp=flow.sbp.split(0),
)

m.to_global(placement=flow.placement("cpu", ranks=[0]), sbp=flow.sbp.broadcast)
m.load_state_dict(global_state_dict)
print(m.state_dict())
```

使用 2 卡运行上面的代码，可以看到，我们自己构造的字典中的全局张量，已经被加载到 m Module 中：

```
OrderedDict([('weight', tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]], placement=oneflow.placement(type="cpu", ranks=[0]), sbp=(oneflow.sbp.broadcast,), dtype=oneflow.float32,
       requires_grad=True)), ('bias', tensor([0., 0., 0., 0., 0., 0.], placement=oneflow.placement(type="cpu", ranks=[0]), sbp=(oneflow.sbp.broadcast,),
       dtype=oneflow.float32, requires_grad=True))])
```


## 一个完整示例

上面，我们演示了如何分片保存和加载模型。在这一部分，提供一份完整的代码参考，下面的例子为 4 个 ranks 上的流水并行，模拟了模型分片保存和加载的过程。

```python
import os
import numpy as np

import oneflow as flow

model_tensor_placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
# model_file_placement 为存储模型分片的设备的 placement，表示在 Rank 2 和 Rank 3 上可为 None。
model_file_placement = flow.placement("cpu", ranks=[0, 1])
P0 = flow.placement(model_tensor_placement.type, ranks=[0])
P1 = flow.placement(model_tensor_placement.type, ranks=[1])
P2 = flow.placement(model_tensor_placement.type, ranks=[2])
P3 = flow.placement(model_tensor_placement.type, ranks=[3])
BROADCAST = flow.sbp.broadcast

def get_sbp(state_dict, tensor):
    if tensor is state_dict["System-Train-TrainStep"]:
        return BROADCAST
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.weight"]:
        return flow.sbp.split(1)
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.bias"]:
        return BROADCAST
    return flow.sbp.split(0)

class Stage0Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = flow.nn.Linear(16, 8)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

class Stage1Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = flow.nn.Linear(8, 4)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

class Stage2Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = flow.nn.Linear(4, 2)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

class Stage3Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = flow.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# 模拟 4 个 ranks 上的流水并行
class PipelineModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_stage0 = Stage0Module()
        self.m_stage1 = Stage1Module()
        self.m_stage2 = Stage2Module()
        self.m_stage3 = Stage3Module()

        self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
        self.m_stage1.to_global(placement=P1, sbp=BROADCAST)
        self.m_stage2.to_global(placement=P2, sbp=BROADCAST)
        self.m_stage3.to_global(placement=P3, sbp=BROADCAST)

    def forward(self, x):
        out_stage0 = self.m_stage0(x)

        in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
        out_stage1 = self.m_stage1(in_stage1)

        in_stage2 = out_stage1.to_global(placement=P2, sbp=BROADCAST)
        out_stage2 = self.m_stage2(in_stage2)

        in_stage3 = out_stage2.to_global(placement=P3, sbp=BROADCAST)
        out_stage3 = self.m_stage3(in_stage3)

        return out_stage3

class PipelineGraph(flow.nn.Graph):
    def __init__(self, module_pipeline):
        super().__init__()
        self.module_pipeline = module_pipeline
        self.module_pipeline.m_stage0.config.set_stage(0, P0)
        self.module_pipeline.m_stage1.config.set_stage(1, P1)
        self.module_pipeline.m_stage2.config.set_stage(2, P2)
        self.module_pipeline.m_stage3.config.set_stage(3, P3)
        self.config.set_gradient_accumulation_steps(2)
        self.add_optimizer(
            flow.optim.SGD(self.module_pipeline.parameters(), lr=0.001)
        )

    def build(self, x):
        out = self.module_pipeline(x)
        out = out.sum()
        out.backward()
        return out

def train_with_graph(call_cnt=0, state_dict_dir=None, last_state_dict=None):
    # 形状为 [2, 16] 的固定输入张量
    x = flow.tensor(
        [
            [
                0.4286,
                0.7402,
                0.4161,
                0.6103,
                0.7394,
                1.1330,
                -0.2311,
                -0.1013,
                0.8537,
                0.9757,
                -0.9842,
                0.3839,
                -0.5551,
                -0.8832,
                0.7820,
                0.7421,
            ],
            [
                -0.1581,
                -1.0319,
                1.8430,
                0.3576,
                0.7288,
                -0.6912,
                0.9966,
                1.0840,
                -1.1760,
                1.5683,
                -0.2098,
                -1.6439,
                -2.7049,
                0.1949,
                1.6377,
                0.0745,
            ],
        ],
        dtype=flow.float32,
        placement=P0,
        sbp=BROADCAST,
    )

    module_pipeline = PipelineModule()
    graph_model = PipelineGraph(module_pipeline)
    cur_rank = flow.env.get_rank()

    if call_cnt == 1:
        if cur_rank in model_file_placement.ranks:
            local_state_dict = flow.load(state_dict_dir)
        else:
            local_state_dict = None

        # 使用 sbp=get_sbp 处理特殊的键
        global_state_dict = flow.utils.global_view.to_global(
            local_state_dict, placement=model_file_placement, sbp=get_sbp,
        )
        graph_model.load_state_dict(global_state_dict)

    graph_model(x)
    state_dict = graph_model.state_dict()

    if call_cnt == 0:
        model_file_state_dict = flow.utils.global_view.to_global(
            state_dict, placement=model_file_placement, sbp=get_sbp,
        )
        if flow.env.get_rank() in model_file_placement.ranks:
            flow.save(
                flow.utils.global_view.to_local(model_file_state_dict),
                state_dict_dir,
            )

if __name__=="__main__":
    rank_id = flow.env.get_rank()
    # 保存路径，一个 rank 对应一个路径。
    state_dict_dir = "./graph_save_load_global_" + str(rank_id)
    # 保存模型
    train_with_graph(0, state_dict_dir)
    # 加载模型
    train_with_graph(1, state_dict_dir)

```

## 结语

上文介绍了：

- 大规模模型分片存储的必要性；
- OneFlow 提供的模型分片保存和加载接口介绍；
- 一个完整的代码例子演示如何完成大模型分片存储；

本文从简单介绍大规模模型分片存储开始，最终演示了 OneFlow 的如何做模型分片保存和加载的过程，后续 OneFlow 的大模型分片存储的接口还会不断完善。
