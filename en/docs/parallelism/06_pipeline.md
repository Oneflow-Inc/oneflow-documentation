# PIPELINING PARALLELISM

We have introduced the characteristics of pipelining parallelism in [COMMON DISTRIBUTED PARALLEL STRATEGIES](./01_introduction.md).

From OneFlow's [global view](./03_consistent_tensor.md), pipelining can be achieved by simply setting the placement attribute of Tensor.

The following code is a simple example that will run the network in [QUICKSTART](../basics/01_quickstart.md) with pipelining parallelism. `nn.Flatten`, `nn.Linear(28*28, 512)` and `nn.ReLU()` run on GPU0, and the rest layers of the network run on GPU1.

??? code
    ```python
    import oneflow as flow

    BATCH_SIZE = 16
    BROADCAST = [flow.sbp.broadcast]
    P0 = flow.placement("cuda", [0])
    P1 = flow.placement("cuda", [1])

    class Stage0Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = flow.nn.Flatten()
            self.linear0 = flow.nn.Linear(28*28, 512)
            self.relu0 = flow.nn.ReLU()

        def forward(self, x):
            out = self.flatten(x)
            out = self.linear0(out)
            out = self.relu0(out)
            return out

    class Stage1Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = flow.nn.Linear(512, 512)
            self.relu1 = flow.nn.ReLU()
            self.linear2 = flow.nn.Linear(512, 10)
            self.relu2 = flow.nn.ReLU()

        def forward(self, x):
            out = self.linear1(x)
            out = self.relu1(out)
            out = self.linear2(out)
            out = self.relu2(out)
            return out

    class PipelineModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.m_stage0 = Stage0Module()
            self.m_stage1 = Stage1Module()

            self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
            self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

        def forward(self, x):
            out_stage0 = self.m_stage0(x)
            in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
            out_stage1 = self.m_stage1(in_stage1)
            return out_stage1

    module_pipeline = PipelineModule()
    sgd = flow.optim.SGD(module_pipeline.parameters(), lr=0.001)

    class PipelineGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.module_pipeline = module_pipeline
            self.module_pipeline.m_stage0.config.set_stage(stage_id=0, placement=P0)
            self.module_pipeline.m_stage1.config.set_stage(stage_id=1, placement=P1)
            self.loss_fn = flow.nn.CrossEntropyLoss()
            self.config.set_gradient_accumulation_steps(2)
            self.add_optimizer(sgd)

        def build(self, x, y):
            out = self.module_pipeline(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            return loss

    graph_pipeline = PipelineGraph()

    x = flow.randn(BATCH_SIZE, 1, 28, 28)
    x = x.to_global(P0, BROADCAST)
    y = flow.randint(0, 10, (BATCH_SIZE,))
    y = y.to_global(P1, BROADCAST)

    for i in range(20):
        loss = graph_pipeline(x, y)
        print(loss.to_local())
    ```

When the code above is saved as a script (`pipeline.py`), it can be then launched by the [launch module](./04_launch.md):


```shell
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./pipeline.py
```

## More Details

### Setting `placement` and `SBP`

Setting up the `placement` and `SBP` at the begining:

```python
BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", [0])
P1 = flow.placement("cuda", [1])
```

`P0` and `P1` represent the 0th GPU and the 1st GPU on the 0th machine respectively.

By calling [nn.Module.to_global](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.Module.html?highlight=nn.Module) or [Tensor.to_global](https://oneflow.readthedocs.io/en/master/tensor.html?highlight=to_global#oneflow.Tensor.to_global), the model or tensor will be distributed to the devices specified before, breaking a network into stages.

Here we define a `PipelineModule` that specifically sets the pipeline for each stage.


```python
    class PipelineModule(flow.nn.Module):
        def __init__(self):
            #...
            
            self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
            self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

        def forward(self, x):
            out_stage0 = self.m_stage0(x)
            in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
            out_stage1 = self.m_stage1(in_stage1)
            return out_stage1
```

### Transforming the Local Tensor to the Global Tensor

The example uses randomly generated data as input.


```python
    x = flow.randn(BATCH_SIZE, 1, 28, 28)
    x = x.to_global(P0, BROADCAST)
```

The `launch` will start two processes when you launch the training by the `launch` module because the command-line parameter is `--nproc_per_node 2`. Both processes will execute the code in the script.

The statement `x = flow.randn(BATCH_SIZE, 1, 28, 28)` returns the Local Tensor (the local data only valid in current process). when running `x = x.to_global(P0, BROADCAST)`, OneFlow will automatically integrate the Local Tensor of all processes into the Global Tensor.


In practice, each computing device can load data locally, and then convert the Local Tensor to the Global Tensor via `to_global`.


### Stage ID and Settings for Gradient Accumulation

We can call the method [config.set_stage](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.graph.block_config.BlockConfig.set_stage.html) of Module Config to set Stage ID and related Placement. The Stage ID are numbered from 0.

We can call the method [config.set_gradient_accumulation_steps](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.graph.graph_config.GraphConfig.set_gradient_accumulation_steps.html#oneflow.nn.graph.graph_config.GraphConfig.set_gradient_accumulation_steps) to set the step size of gradient accumulation.

The information needed to implement micro-batch in pipelining parallelism can be obtained by these two configurations.


```python
    self.module_pipeline.m_stage0.config.set_stage(stage_id=0, placement=P0)
    self.module_pipeline.m_stage1.config.set_stage(stage_id=1, placement=P1)
    self.config.set_gradient_accumulation_steps(2)
```
