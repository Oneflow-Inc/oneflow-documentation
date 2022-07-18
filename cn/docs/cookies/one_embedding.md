# 大规模 Embedding 方案： OneEmbedding

Embedding 是推荐系统的重要组件，也扩散到了推荐系统外的许多领域。各个框架都提供了进行 embedding 的基础算子，比如 OneFlow 中的 `flow.nn.Embedding`：

```python
import numpy as np
import oneflow as flow
indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
embedding = flow.nn.Embedding(10, 3)
y = embedding(indices)
```

OneFlow 为了解决大规模深度推荐系统的问题，还提供了大规模 Embedding 的方案：OneEmbedding。

与普通的算子相比，OneEmbedding 有以下特点：

1. 灵活的分层存储，支持将 Embedding table 放置在 GPU 显存、 CPU 内存或者 SSD 上面，允许使用高速设备作为低速设备的缓存，实现速度与容量的兼顾。

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

点击 [make_table_options](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.make_table_options) 及 [make_uniform_initializer](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.make_uniform_initializer) 可以查看有关它们的更详细说明。

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

这里通过调用 `make_cached_ssd_store_options`，选择将词表存储在 SSD 中，并且使用 GPU 作为高速缓存。具体参数的意义可以参阅 [make_cached_ssd_store_options API 文档](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.make_cached_ssd_store_options)。

此外，还可以选择使用纯 GPU 存储；或者使用 CPU 内存存储词表、但用 GPU 做高速缓存。具体可以分别参阅 [make_device_mem_store_options](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.make_device_mem_store_options) 及 [make_cached_host_mem_store_options](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.make_cached_host_mem_store_options)。

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

如果同时创建了两个 OneEmbedding，在实例化时需要设置不同的 name 和 persistent path 参数。 更详细的信息，可以参阅 [one_embedding.MultiTableEmbedding](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.MultiTableEmbedding)

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

**特征 ID 不能重复**

制作数据集的 OneEmbedding 用户需要格外注意：使用 `MultiTableEmbedding` 同时创建多个表时，多个 Embedding Table 仅初始化参数不同，其他参数一致，此时，**多个表中的特征 ID 不能重复** 。

**多表查询**

如果使用 `MultiTableEmbedding` 只配置了一个表，则查询方式与普通的 Embedding 查询方式没有区别，直接调用，并传递特征 ID 即可，如 `embedding_lookup(ids)`。

如果使用 `MultiTableEmbedding` 配置了多个表，则对于某个特征 ID，需要指明在哪个表中查询，有两种方式指明：

方法一：传递一个形状为 `(batch_size, 词表数目)` 的 `ids` 用于查询，则这个 `ids` 的列，依次对应一个词表。

比如：
```python
ids = np.array([[488, 333, 220], [18, 568, 508]], dtype=np.int64)
# 表示在第 0 个表中查询 `[[488], [18]]`，第 1 个表中查询 `[[333], [568]]`，第 2 个表中查询 `[[220], [508]]` 对应的特征向量。
embedding_lookup(ids)
```

方法二：传递 `ids` 参数的同时，再传递一个 `table_ids` 参数，它的形状与 `ids` 完全相同，在 `table_ids` 中指定表的序号。

比如：
```python
ids = np.array([488, 333, 220, 18, 568, 508], dtype=np.int64)
# table_ids的shape与ids保持一致
table_ids = np.array([0, 1, 2, 0, 1, 2])
# 表示在第 0 个表中查询 `488, 18`，第 1 个表中查询 `333, 568`，第 2 个表中查询 `220, 508` 对应的特征向量。
embedding_lookup(ids, table_ids)
```

更详细的说明，可以参阅 [MultiTableEmbedding.forward](https://oneflow.readthedocs.io/en/master/one_embedding.html#oneflow.one_embedding.MultiTableEmbedding.forward)

### 如何选择合适的存储配置

OneEmbedding 提供了三种存储选项配置，分别是纯 GPU 存储， 存储在 CPU 内存中并使用 GPU 显存作为高速缓存和存储在 SSD 中，并使用 GPU 显存作为高速缓存。

- 纯 GPU 存储

    当词表大小小于 GPU 显存时，将全部词表放在 GPU 显存上是最快的，此时推荐选择纯 GPU 存储配置。
    
- 存储在 CPU 内存中，并使用 GPU 显存作为高速缓存

    当词表大于 GPU 显存，但是小于 CPU 内存时，推荐词表存储在 CPU 内存中，并使用 GPU 显存作为高速缓存。
    
- 存储在 SSD 中，并使用 GPU 显存作为高速缓存

    当词表大小既大于 GPU 显存，也大于系统内存时，如果有高速的 SSD，可以选择将词表存储在 SSD 中，并使用 GPU 显存作为高速缓存。在此情况下，训练过程中会对存储的词表进行频繁的数据读写，因此 `persistent_path` 所设置路径下的文件随机读写速度对整体性能影响很大。强烈推荐使用高性能的 SSD，如果使用普通磁盘，会对性能有很大负面影响。


### 分布式训练

OneEmbedding 同 OneFlow 的其它模块类似，都原生支持分布式扩展。用户可以参考 [#dlrm](扩展阅读：DLRM) 中的 README， 启动 DLRM 分布式训练。还可以参考 [Global Tensor](../parallelism/03_consistent_tensor.md) 了解必要的前置知识。

使用 OneEmbedding 模块进行分布式扩展，要注意：

- 目前 OneEmbedding 只支持放置在全部设备上，并行度需和 world size 一致。比如，在 4 卡并行训练时，词表的并行度必须为 4，暂不支持网络使用 4 卡训练但词表并行度为 2 的场景。
- `store_options` 配置中参数 `persistent_path` 指定存储的路径。在并行场景中，它既可以是一个表示路径的字符串，也可以是一个 `list`。若配置为一个代表路径的字符串，它代表分布式并行中各 rank 下的根目录。OneFlow 会在这个根路径下，依据各个 rank 的编号创建存储路径，名称格式为 `rank_id-num_rank`。若`persistent_path` 是一个 `list`，则会依据列表中的每项，为 rank 单独配置。
- 在并行场景中，`store_options` 配置中的 `capacity` 代表词表总容量，而不是每个 rank 的容量。`cache_budget_mb` 代表每个 GPU 设备的显存。

### 基础概念
这一章介绍几个概念在OneEmbedding语境中的含义。
#### EmbeddingTable与MultiTableEmbedding
在OneEmbedding中嵌入表（EmbeddingTable）既可以是一个从索引映射到稠密向量的查找表，也可以是一个键值对（key value pair）查找表。在一些场景中，可能会用到多个嵌入表，比如在推荐系统中，每一个特征都对应一张嵌入表。如果在模型中使用多张嵌入表，查表的性能一般比较低，OneEmbedding推荐把多张表合成一张表使用的做法，只需要保证多张表的id或者key不重复即可，这里被称为多表嵌入（MultiTableEmbedding）。
用户在使用MultiTableEmbedding的时候，可能与普通的EmbeddingTable有不同，比如：
- 制作数据集的时候要注意不同表的id或者key不能重复；
- 不同表所期待的初始化方式可能不同。

#### 分层存储

#### 分层存储
随着嵌入表规模的扩大，在一些情况下嵌入表已经大到无法被设备内存、或者主机内存完整装入，OneEmbedding支持灵活的分层存储，支持将 Embedding table 放置在 GPU 显存、 CPU 内存或者 SSD 上面，允许使用高速设备作为低速设备的缓存，实现速度与容量的兼顾。目前OneEmbedding开放了三种分层存储模式：
- `device_mem`：如果嵌入表还能够被设备内存完整装入，而且设备上还有足够的内存供网络模型的其他部分使用，这就是一种最高效的模式，可以使用`oneflow.one_embedding.make_device_mem_store_options`进行配置。
- `cached_host_mem`：如果嵌入表无法被完整的装入设备内存，但主机内存足够大，OneEmbedding支持将主机内存作为主要的存储介质，设备内存作为一级缓存动态的存储高频部分的词表，使用`oneflow.one_embedding.make_cached_host_mem_store_options`进行配置。这种模式的性能接近略低于`device_mem`模式。
- `cached_ssd`：如果主机内存也不够大，OneEmbedding支持将高速SSD作为主要的存储介质，设备内存作为一级缓存动态的存储高频部分的词表，使用`oneflow.one_embedding.make_cached_ssd_store_options`进行配置。这里强调使用高速SSD，是从性能上考虑。

#### 持久化存储
训练好的嵌入表需要被持久化的保存下来，在配置分层存储时，会被要求配置持久化存储目录（persistent_path)，OneEmbedding将模型数据保存到这个目录中，不过保存方式和其他variable有不同。

我们先从一般的模型保存开始说起，模型的保存一般是保存的state_dict，如下面的操作从module中提取state_dict并保存到指定目录`saved_snapshot`：

```python
flow.save(module.state_dict(), "saved_snapshot", global_dst_rank=0)
```

假设module中含有OneEmbedding，让我们看看里面存了啥？

```python
>>> import oneflow as flow
loaded library: /lib/x86_64-linux-gnu/libibverbs.so.1
>>> state_dict = flow.load("saved_snapshot")
>>> state_dict.keys()
odict_keys(['bottom_mlp.linear_layers.weight_0', 'bottom_mlp.linear_layers.bias_0', 'bottom_mlp.linear_layers.weight_1', 'bottom_mlp.linear_layers.bias_1', 'bottom_mlp.linear_layers.weight_2', 'bottom_mlp.linear_layers.bias_2', 'embedding.one_embedding.OneEmbedding', 'top_mlp.linear_layers.weight_0', 'top_mlp.linear_layers.bias_0', 'top_mlp.linear_layers.weight_1', 'top_mlp.linear_layers.bias_1', 'top_mlp.linear_layers.weight_2', 'top_mlp.linear_layers.bias_2', 'top_mlp.linear_layers.weight_3', 'top_mlp.linear_layers.bias_3', 'top_mlp.linear_layers.weight_4', 'top_mlp.linear_layers.bias_4'])
>>> state_dict['embedding.one_embedding.OneEmbedding']
'2022-04-15-22-53-04-270525'
```

其中`embedding.one_embedding.OneEmbedding`就是OneEmbedding的模型，但是里面存的是一个字符串'2022-04-15-22-53-04-270525'，我们去persistent_path看看：

```bash
$ tree -d persistent_path
persistent_path
├── 0-4
│   ├── keys
│   ├── snapshots
│   │   └── 2022-04-15-22-53-04-270525
│   └── values
├── 1-4
│   ├── keys
│   ├── snapshots
│   │   └── 2022-04-15-22-53-04-270525
│   └── values
├── 2-4
│   ├── keys
│   ├── snapshots
│   │   └── 2022-04-15-22-53-04-270525
│   └── values
└── 3-4
    ├── keys
    ├── snapshots
    │   └── 2022-04-15-22-53-04-270525
    └── values
```

发现里面有四个子目录，`0-4` `1-4` `2-4` `3-4`，这是为4个GPU设备分别准备的4个目录，这四个目录的`snapshots`中都有一个`2022-04-15-22-53-04-270525`目录。这就是持久化保存的目录内容。

## 扩展阅读：DLRM    

本文展示了如何快速上手 OneEmbedding。
OneFlow 模型仓库中准备了关于 OneEmbedding 在 DLRM 任务的实际例子，可供参考：https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/dlrm
