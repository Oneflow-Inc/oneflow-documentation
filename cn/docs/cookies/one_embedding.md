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
import numpy as np

tables = [
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.1, high=0.1)
    ),
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.05, high=0.05)
    ),
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.15, high=0.15)
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


## OneEmbedding 特点说明

### 特征 ID 动态扩容

OneEmbedding 支持动态插入新的特征 ID，只要存储介质的容量足够，特征 ID 的数目是没有上限的。这也是为什么在使用 `make_table_options` 时，只需要指定初始化方式，不需要指定特征 ID 的总数目（词表行数）。

### 特征 ID 与多表查询

**特征ID 不能重复**

制作数据集的 OneEmbedding 用户需要格外注意：使用 `MultiTableEmbedding` 同时创建多个表时，**多个表中的特征 ID 不能重复** 。

**多表查询**

如果使用 `MultiTableEmbedding` 只配置了一个表，则查询方式与普通的 Embedding 查询方式没有区别，直接调用，并传递特征 ID 即可，如 `embedding_lookup(ids)`。

如果使用 `MultiTableEmbedding` 配置了多个表，则对于某个特征 ID，需要指明在哪个表中查询，有两种方式指明：

方法一：传递一个形状为 `(batch_size, 表格数目)` 的 `ids` 用于查询，则这个 `ids` 的列，依次对应一个表格。

比如：`ids = np.array([[488, 333, 220], [18, 568, 508]], dtype=np.int64)`，表示在第 0 个表中查询 `[[488], [18]]`，第 1 个表中查询 `[[333], [568]]`，第 2 个表中查询 `[[220], [508]]` 对应的特征向量。

方法二：传递 `ids` 参数的同时，再传递一个 `table_ids` 参数，它的形状与 `ids` 完全相同，在 `table_ids` 中指定表的序号。

比如：`ids = np.random.array([488, 333, 220, 18, 568, 508], dtype=np.int64)`，`table_ids = np.random.array([0, 1, 2, 0, 1, 2])`，然后调用： `embedding_lookup(ids, table_ids)`，则表示在第 0 个表中查询 `488, 18`，第 1 个表中查询 `333, 568`，第 2 个表中查询 `220, 508` 对应的特征向量。

更详细的说明，可以参阅 [MultiTableEmbedding.forward]()

### 如何选择合适的存储配置

OneEmbedding 提供了三种存储选项配置：

- 纯 GPU 存储
- 存储在 CPU 内存中，并使用 GPU 显存作为高速缓存
- 存储在 SSD 中，并使用 GPU 显存作为告诉缓存

用户可以根据实际情况选择最优的方案，一般选择的依据是：

- 当词表大小小于 GPU 显存时，将全部词表放在 GPU 显存上是最快的，此时推荐选择纯 GPU 存储配置。
- 当词表大于 GPU 显存，但是小于 CPU 内存时，推荐词表存储在 CPU 内存中，并使用 GPU 显存作为高速缓存。
- 当词表大小既大于 GPU 显存，也大于系统内存时，如果有高速的 SSD，可以选择将词表存储在SSD中，并使用 GPU 显存作为高速缓存。在此情况下，训练过程中会对存储的词表进行频繁的数据读写，因此 `persistent_path` 所设置路径下的文件随机读写速度对整体性能影响很大。强烈推荐使用高性能的 SSD，如果使用普通磁盘，会对性能有很大负面影响。

### 分布式训练

OneEmbedding 同 OneFlow 的其它模块类似，都原生支持分布式扩展。用户可以参考 [#dlrm](扩展阅读：DLRM) 中的 README， 启动 DLRM 分布式训练。还可以参考 [Global Tensor](../parallelism/03_consistent_tensor.md) 了解必要的前置知识。

使用 OneEmbedding 模块进行分布式扩展，要注意：

- 目前 OneEmbedding 只支持放置在全部设备上，并行度需和 world size 一致。比如，在 4 卡并行训练时，词表的并行度必须为 4，暂不支持网络使用 4 卡训练但词表并行度为 2 的场景。
- `store_options` 配置中参数 `persistent_path` 指定存储的路径。在并行场景中，它既可以是一个表示路径的字符串，也可以是一个 `list`。若配置为一个代表路径的字符串，它代表分布式并行中各 rank 下的根目录。OneFlow 会在这个根路径下，依据各个 rank 的编号创建存储路径，名称格式为 `rank_id-num_rank`。若`persistent_path` 是一个 `list`，则会依据列表中的每项，为 rank 单独配置。
- 在并行场景中，`store_options` 配置中的 `capacity` 代表词表总容量，而不是每个 rank 的容量。`cache_budget_mb` 代表每个 GPU 设备的显存。

## 扩展阅读：DLRM    

本文展示了如何快速上手 OneEmbedding。

OneFlow 模型仓库中准备了关于 OneEmbedding 在 DLRM 任务的实际例子，可供参考：https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/dlrm
