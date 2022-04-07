# 大规模 Embedding 方案： OneEmbedding

Embedding 已经成为推荐系统的基本操作，也扩散到了推荐系统外的许多领域。各个框架都提供了进行 embedding 的基础算子，比如，可以使用 OneFlow 中的 `flow.nn.Embedding`：

```python
import numpy as np
import oneflow as flow
indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
embedding = flow.nn.Embedding(10, 3)
y = embedding(indices)
```

OneFlow 为了解决大规模深度推荐系统的问题，还提供了大规模 Embedding 的方案：OneEmbedding。

与普通的算子相比，OneEmbedding 有以下特点：

1. 灵活的分层存储，支持将 Embedding table 放置在 GPU显存、CPU内存 、或者 SSD 上面，使用高速设备作为低速设备的缓存，实现速度与容量的兼顾。

2. 支持动态扩容。


## 快速上手 OneEmbedding

我们将展示一个快速上手 OneEmbedding 的例子，它包括以下几个基本步骤：

- 使用 `make_table_options` 配置词表
- 配置词表的存储属性
- 实例化 Embedding
- 构造 Graph 进行训练



### 使用 `make_table_options` 配置词表

以下代码，导入相关包，并使用 `make_table_options` 配置词表。OneEmbedding 支持同时创建多个词表，以下代码配置了三个词表。

```python
import oneflow as flow
import oneflow.nn as nn

tables = [
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.1, height=0.1)
    ),
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.05, height=0.05)
    ),
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.15, height=0.15)
    ),
]
```

配置词表时需要指定初始化的方式，以上词表均采用 `uniform` 方式初始化。配置词表的结果保存在 `tables` 变量中。

点击 [make_table_options]() 及 [make_uniform_initializer]() 可以查看有关它们的更详细说明。

### 配置词表的存储属性

接着运行以下代码，用于配置词表的存储属性：

```python
store_options = flow.one_embedding.make_cached_ssd_store_options(
    cache_budget_mb=8142,
    persistent_path="/your_path_to_ssd", 
    capacity=40000000,
    size_factor=1,   			
    physical_block_size=512
)
```

这里通过调用 `make_cached_ssd_store_options`，选择将词表存储在 SSD 中，并且使用 GPU 作为高速缓存。具体参数的意义可以参阅 [make_cached_ssd_store_options API 文档]()。

此外，还可以选择使用纯 GPU 存储；或者使用 CPU 内存存储词表、但用 GPU 做高速缓存。具体可以分别参阅 [make_cached_ssd_store_options]() 及 [make_cached_host_mem_store_option]()。

### 实例化 Embedding

以上配置完成后，使用 `MultiTableEmbedding` 可以得到实例化的 Embedding 层。

```python
embedding_size = 128
embedding = flow.one_embedding.MultiTableEmbedding(
    name="my_embedding",
    embedding_dim=embedding_size,
    dtype=flow.float,
    key_type=flow.int64,
    tables=tables,
    store_options=store_options,
)

embedding.to("cuda")
```

其中，`tables` 是之前通过 `make_table_options` 配置的词表属性，`store_options` 是之前配置的存储属性，`embedding_dim` 是特征维度，`dtype` 是特征向量的数据类型，`key_type` 是特征 ID 的数据类型。

更详细的信息，可以参阅 [one_embedding.MultiTableEmbedding]()

### 使用 Graph 训练

目前 OneEmbedding 仅支持在 Graph 模式下使用。

在以下的例子中，我们构建了一个简单的 Graph 类，它包括了 `embedding` 和 `mlp` 两层。

```python
num_tables = 3
mlp = flow.nn.FusedMLP(
    in_features=embedding_size * num_tables,
    hidden_features=[512, 256, 128],
    out_features=1,
    skip_final_activation=True,
)
mlp.to("cuda")

class TrainGraph(flow.nn.Graph):
    def __init__(self,):
        super().__init__()
        self.embedding_lookup = embedding
        self.mlp = mlp
        self.add_optimizer(
            flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.0)
        )
        self.add_optimizer(
            flow.optim.SGD(self.mlp.parameters(), lr=0.1, momentum=0.0)
        )
    def build(self, ids):
        embedding = self.embedding_lookup(ids)
        loss = self.mlp(flow.reshape(embedding, (-1, num_tables * embedding_size)))
        loss = loss.sum()
        loss.backward()
        return loss
```

然后就可以实例化 Graph，开始训练了。

```python
ids = np.random.randint(0, 1000, (100, num_tables), dtype=np.int64)
ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
graph = TrainGraph()
loss = graph(ids_tensor)
print(loss)
```

关于 Graph 的使用，可以参阅 [静态图模块 nn.Graph](../basics/08_nn_graph.md) 获取更详细信息。

### 存储和加载

使用 `MultiTableEmbedding.save_snapshot` 方法可以保存词表：

```python
embedding.save_snapshot('./my_snapshot')
```

如果想从已保存的快照中加载词表，则应该调用 `MultiTableEmbedding.load_snapshot`。

更详细的信息请参阅 [MultiTableEmbedding.save_snapshot]() 及 [MultiTableEmbedding.load_snapshot]()。

## 动态插入新的特征 ID

需要重点说明，OneEmbedding 支持动态插入新的特征 ID，只要存储介质的容量足够，特征 ID 的数目是没有上限的。具体体现在，进行查询时，特征 ID 可以超越创建词表时的范围。

这也是为什么在使用 `make_table_options` 时，只需要指定初始化方式，不需要指定特征 ID 的总数目（词表行数）。

## 扩展阅读：DLRM    

本文展示了如何快速上手 OneEmbedding。

OneFlow 模型仓库中准备了关于 OneEmbedding 在 DLRM 任务的实际例子，可供参考：https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/dlrm
