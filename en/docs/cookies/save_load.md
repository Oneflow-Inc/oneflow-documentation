# OneFlow's Distributed Saving and Loading of Large Models

## About the distributed saving of large-scale models

When a model is relatively small, such as less than 100G, it is possible to save it in a node. But when the parameters of a model are relatively large, the number of samples required at this time is also larger, and the model produced by dump after training will also be very large, which can't be saved by a single node. For example, Megatron Turing Natural Language Generation Model (MT-NLG) driven by DeepSpeed and Megatron has 530 billion parameters, which is the largest and most powerful monolithic Transformer language model trained so far. So, distributed save and load will be needed to support such a large-scale language model rather than single-node memory. In addition, in other CV, search, recommendation and advertising scenarios, the increase of sample size and model complexity will bring more difficulties in model storage.

This paper will introduce OneFlow's distributed saving and loading of large-scale model and its usage.

## OneFlow's distributed saving and loading of models

The implementation of OneFlow's distributed saving and loading of large model is based on the concept of [Global View](https://docs.oneflow.org/master/cookies/global_tensor.html), which not only uses Placement and SBP to complete the segmentation of model files (represented by state dict below) on various physical devices, but also is suitable for scenarios when the model is too large to be accommodated on the memory or video memory of a single device.

### About the flow.utils.global_view.to_global() interface

In order to better understand the following two parts: saving and loading the model, let's first analyze the `flow.utils.global_view.to_global()` interface and its implementation ideas. Different from the existing [Tensor.to_global()](https://oneflow.readthedocs.io/en/master/generated/oneflow.Tensor.to_global.html?highlight=to_global%28%29) mode (which can handle common Tensor), `flow.utils.global_view.to_global()` interface provides multiple types of input support, including None, Tensor, List, Tuple, state dict of nn.Module, state dict of nn.Graph, and any combination of several types. Besides, it converts the input Tensor in List/Tuple/Dict to Global Tensor. It is worth noting that the SBP in the inputting parameter allows the user to customize a `(x, tensor) -> sbp` function to solve the needs of different Tensors corresponding to different SBPs.

In addition, corresponding to to_global() is the `flow.utils.global_view.to_local()` interface. You can refer to the API documentation for a more [detailed introduction](https://oneflow.readthedocs.io/en/master/utils.global_view.html) about to_global() and to_local(). In the [implementation](https://github.com/Oneflow-Inc/oneflow/blob/master/python/oneflow/utils/global_view/to_global.py) of `flow.utils.global_view.to_global()`, multiple input types are available, which are applicable to the existing `Tensor.to_global()` interface. The overall idea of the implementation is roughly to check the input and broadcasting (null) structure, traverse the nodes, call the callback function and return the result after to_global().

Going back to what we are concerned: how does this interface realize distributed saving and loading? For example, for model parallelism/pipeline parallelism, the parameters of the model are scattered on multiple ranks. Before saving the model, using `flow.utils.global_view.to_global()` to convert each Tensor in the state dict to Global Tensor on the specified Placement. As the type of SBP is `flow.sbp.split`, it can set the split on a specific dimension. Likewise, models can also be loaded by Split. Of course, SBP can also be broadcast, supporting different combinations of SBP and Placement. In this way, the problem of distributed storage of super-large models is solved very well.

### Saving models

After an overview of the `flow.utils.global_view.to_global()` interface, this section will demonstrate how to save the model distributively. The code is as follows:

```python

# Customize the get_sbp function.
def get_sbp(state_dict, tensor):
    if tensor is state_dict["System-Train-TrainStep"]:
        return flow.sbp.broadcast
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.weight"]:
        return flow.sbp.split(1)
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.bias"]:
        return flow.sbp.broadcast
    return flow.sbp.split(0)

model_file_state_dict = flow.utils.global_view.to_global(
    state_dict, placement=model_file_placement, sbp=get_sbp, 
    ) # Use sbp=get_sbp to handle special keys, and common SBPs are also available.

rank_id = flow.env.get_rank()
# Save the path of distributed model, and a rank corresponds to a path.
state_dict_dir = "./graph_save_load_global_" + str(rank_id)

if flow.env.get_rank() in model_file_placement.ranks:
    flow.save(
        flow.utils.global_view.to_local(model_file_state_dict),
        state_dict_dir,
    )
```

First, convert the original model (state_dict) to the model file's Placement and SBP, and model_file_place is the array of devices to the model waiting to be saved distributively, that is, distribute state dict to model_file_place by split (0). The reason for customizing the get_sbp function here is that the user can pass in an ` (x, tensor)-> sbp ` function to address the need for a particular tensor to correspond to a different SBP. For example (the current example is based on the Graph mode), for `state_dict["System-Train-TrainStep"]`, a Tensor whose shape is [1], we can't distributed by split (0), and SBP can choose broadcast. `state_dict["module_pipeline"]["m_stage3.linear.weight"]` can only be distributed in the first dimension. For `state_dict["module_pipeline"]["m_stage3.linear.bias"]`, non-distributed small tensor (s), SBP can choose broadcast. In this way, it allows user to DIY SBP processing, which is more flexible.

In later processing, using the `flow.utils.global_view.to_local()` interface to get the local component of model_file_state_dict and call [save()](https://oneflow.readthedocs.io/en/master/generated/oneflow.save.html?Highlight=save) to save the model. Among them, state_dict_dir is a directory with device id, so it is necessary to distinguish different devices. It is recommended that a rank correspond to a path, and use rank id to represent the path name 

### Loading models

After saving the model distributively on the specified device, the code for loading the model is as follows:

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

First, using [load()](https://oneflow.readthedocs.io/en/master/generated/oneflow.load.html?Highlight=load) to load the state dict on each device where the distributed model is saved. Correspondingly, you need to convert the state dict on the local rank to the placement and sbp of the model file to obtain global_state_dict. This step should correspond to saving the model, and SBP and Placement are also consistent. Finally, global_state_dict can be successfully loaded into graph_model (nn.Graph). Of course, the processing methods of nn.Module and nn.Graph are consistent.

### Load state dict into nn.Module

In addition to the above two features, OneFlow provides automatic conversion of SBP and Placement when loading state dict into nn.Module. In the following example, we first construct an m (nn.Module) object, then set the SBP of global_state_dict to split(0) and the SBP of m to broadcast. At the same time, the placement has changed, from `placement("cpu", ranks=[0, 1])` to `flow.placement("cpu", ranks=[0])`. At this time, the user does not need other operations, and OneFlow will automatically do the conversion process between SBP and placement.

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

Running the above code with 2 GPUs, you can see that the global tensor in our own dictionary has been loaded into m Module. In addition, the SBP of the tensor that outputs OrderedDict has been automatically converted from split (0) to broadcast, and the shape of 'weight' corresponding to the tensor is also `[6, 2]`, and the shape of 'bias' is `[6]`.

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

## A complete example

Above, we demonstrated how to save and load the model distributively. In this section, a complete code reference is provided. The following example is pipeline parallelism on 4 ranks, which simulates the process of distributed saving and loading of models.

```python
import os
import numpy as np

import oneflow as flow

model_tensor_placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
# model_file_placement is the placement of the device that stores the distributed models, which can be None on Rank 2 and Rank 3.

model_file_placement = flow.placement("cpu", ranks=[0, 1])
P0 = flow.placement(model_tensor_placement.type, ranks=[0])
P1 = flow.placement(model_tensor_placement.type, ranks=[1])
P2 = flow.placement(model_tensor_placement.type, ranks=[2])
P3 = flow.placement(model_tensor_placement.type, ranks=[3])

def get_sbp(state_dict, tensor):
    if tensor is state_dict["System-Train-TrainStep"]:
        return flow.sbp.broadcast
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.weight"]:
        return flow.sbp.split(1)
    if tensor is state_dict["module_pipeline"]["m_stage3.linear.bias"]:
        return flow.sbp.broadcast
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

# Simulate pipeline parallelism on 4 ranks

class PipelineModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_stage0 = Stage0Module()
        self.m_stage1 = Stage1Module()
        self.m_stage2 = Stage2Module()
        self.m_stage3 = Stage3Module()

        self.m_stage0.to_global(placement=P0, sbp=flow.sbp.broadcast)
        self.m_stage1.to_global(placement=P1, sbp=flow.sbp.broadcast)
        self.m_stage2.to_global(placement=P2, sbp=flow.sbp.broadcast)
        self.m_stage3.to_global(placement=P3, sbp=flow.sbp.broadcast)

    def forward(self, x):
        out_stage0 = self.m_stage0(x)

        in_stage1 = out_stage0.to_global(placement=P1, sbp=flow.sbp.broadcast)
        out_stage1 = self.m_stage1(in_stage1)

        in_stage2 = out_stage1.to_global(placement=P2, sbp=flow.sbp.broadcast)
        out_stage2 = self.m_stage2(in_stage2)

        in_stage3 = out_stage2.to_global(placement=P3, sbp=flow.sbp.broadcast)
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
    # Fixed input tensor of shape [2, 16]
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
        sbp=flow.sbp.broadcast,
    )

    module_pipeline = PipelineModule()
    graph_model = PipelineGraph(module_pipeline)
    cur_rank = flow.env.get_rank()

    if call_cnt == 1:
        if cur_rank in model_file_placement.ranks:
            local_state_dict = flow.load(state_dict_dir)
        else:
            local_state_dict = None

        # Use sbp=get_sbp to handle special keys
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
    # Save the path, and a rank corresponds to a path.
    state_dict_dir = "./graph_save_load_global_" + str(rank_id)
    # Save the model
    train_with_graph(0, state_dict_dir)
    # Load the model
    train_with_graph(1, state_dict_dir)

```

## Conclusion

The following contents are introduced above:

- The necessity of distributed saving of large-scale models;
- Introduction of the interface for saving and loading models distributively provided by OneFlow;
- A complete code example demonstrating how to achieve the distributed saving of large models.

This paper starts with a brief introduction of distributed saving of large-scale models, and finally demonstrates how OneFlow does the process of distributed saving and loading of models. In the future, OneFlow will continue to improve the interface of distributed saving of large models.
