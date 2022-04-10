# Large-Scale Embedding Solution: OneEmbedding

Embedding is an important component of recommender system, and it has also spread to many fields outside recommender systems. Each framework provides basic operators for Embedding, for example, `flow.nn.Embedding` in OneFlow:

```python
import numpy as np
import oneflow as flow
indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
embedding = flow.nn.Embedding(10, 3)
y = embedding(indices)
```

OneEmbedding is the large-scale Embedding solution that OneFlow provides to solve the problem of large-scale deep recommender systems. OneEmbedding has the following advantages compared to ordionary opeartors:

1. With Flexible hierarchical storage, OneEmbedding can place the Embedding table on GPU memory, CPU memory or SSD, and allow high-speed devices to be used as caches for low-speed devices to achieve both speed and capacity.

2. OneEmbedding supports dynamic expansion.

## Get Start to OneEmbedding Quickly

The following steps is an example of getting started with OneEmbeeding quickly: 

- Configure Embedding table with `make_table_options`
- Configure the storage attribute of the Embedding table
- Instantiate Embedding
- Construct Graph for training

### Configure Embedding Table with `make_table_options`

By importing relevant package and the following codes, you can configure Embedding table with `make_table_options`.OneEmbedding supports simultaneous creation of multiple Embedding table. The following codes configured three Embedding table.

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

When configuring the Embedding table, you need to specify the initialization method. The above Embedding tables are initialized in the `uniform` method. The result of configuring the Embedding table is stored in the `tables` variable

Click [make_table_options]() and [make_uniform_initializer]() to check more detailed information.

### Configure the Storage Attribute of the Embedding Table

Then run the following codes to configure the storage attribute of the Embedding table:

```python
store_options = flow.one_embedding.make_cached_ssd_store_options(
    cache_budget_mb=8142,
    persistent_path="/your_path_to_ssd", 
    capacity=40000000,
    size_factor=1,   			
    physical_block_size=512
)
```

By calling `make_cached_ssd_store_options` here, you can store Embedding table on SSD and use GPU as cache. For the meaning of specific parameters, please refer to [make_cached_ssd_store_options API 文档]().

In addition, you can use pure GPU as storage; or use CPU memory to store Embedding table, but use GPU as cache. For more details, please refer to [make_cached_ssd_store_options]() and [make_cached_host_mem_store_option]().

### Instantiate Embedding

After the above configuration is completed, you can use `MultiTableEmbedding` to get the instantiated Embedding layer.

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

Among them, `tables` is the Embedding table attribute previously configured by `make_table_options`, `store_options` is the previously configured storage attribute, `embedding_dim` is the feature dimension, `dtype` is the data type of the feature vector, `key_type` is the data type of feature ID.

For more detailes, please refer to [one_embedding.MultiTableEmbedding]().

### Construct Graph for Training

Currently, OneEmbedding is only supported in Graph mode. 

In the following example, we construct a simple Graph class that includes `embedding` and `mlp` layers.

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

Then you can instantiate the Graph and start training.

```python
ids = np.random.randint(0, 1000, (100, num_tables), dtype=np.int64)
ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
graph = TrainGraph()
loss = graph(ids_tensor)
print(loss)
```

For the detailed information on using Graph, please refer to [静态图模块 nn.Graph](../basics/08_nn_graph.md). 

## The Features of OneEmbedding

### Feature ID and Dynamic Insertion

OneEmbedding supports dynamic insertion of new feature ID. As long as the storage medium has sufficient capacity, there is no upper limit on the number of feature IDs. This is why when you use `make_table_options`, you only need to specify the initialization method, not the total number of feature IDs (Embedding table lines).

### Feature ID and Multi-Table Query

**Feature ID cannot be repeated**

OneEmbedding users who make datasets need to pay special attention: When using `MultiTableEmbedding` to create multiple tables at the same time, **feature IDs in multiple tables cannot be repeated**.

**Multi-table query**

The query method is no different from the normal Embedding query method if you only use `MultiTableEmbedding` to configure one table. You can call it directly and pass the feature ID, such as `embedding_lookup(ids)`.

If you use `MultiTableEmbedding` to configure more than one tables, then you need to specify in which to query for a feature ID with the following two methods:

Method 1: Pass an `ids` of shape `(batch_size, 表格数目)`  for query, then the column of this `ids` corresponds to a table in turn.

For example, `ids = np.array([[488, 333, 220], [18, 568, 508]], dtype=np.int64)` means to query `[[488], [18]]` in the zeroth table, `[[333], [568]]` in the first table, and the corresponding feature vector of `[[220], [508]]` in the second table.

Method 2:When passing the `ids` parameter, pass a `table_ids` parameter, which has the exact same shape as `ids`, and specifies the ordinal number of the table in `table_ids`.

For example:
```python
ids = np.array([488, 333, 220, 18, 568, 508], dtype=np.int64)
# table_ids has the exact same shape as `ids`
table_ids = np.array([0, 1, 2, 0, 1, 2])
# This means to query `488, 18` in the zeroth table, `333, 568` in the first table, and the corresponding feature vector of `220, 508` in the second table.
embedding_lookup(ids, table_ids)
```
For more details, please refer to  [MultiTableEmbedding.forward]().

### How to Choose the Proper Storage Configuration 

OneEmbedding provides three storage options configurations,they are pure GPU storage, use CPU memory to store and GPU memory as cache and use SSD to store and GPU memory as cache.

- Pure GPU storage

    When the size of Embedding table is smaller than the GPU memory, it is the fastest to place all the Embedding table on the GPU memory. In this case, it is recommended to select the pure GPU storage configuration.
    
- Use CPU memory to store and GPU memory as cache

    When the size of Embedding table is larger than the GPU memory, but smaller than the CPU memory, it is recommended to store the Embedding table in the CPU memory and use the GPU memory as cache. 
    
- Use SSD to store and GPU memory as cache

    When the size of Embedding table is larger than both the GPU memory and the system memory, if you have a high-speed SSD, you can choose to store the Embedding table in the SSD and use the GPU memory as a cache. In this case, frequent data reading and writing will be performed on the stored vocabulary during the training process, so the random reading and writing speed of files under the path set by `persistent_path` has a great impact on the overall performance. It is strongly recommended to use a high-performance SSD. If you use a normal disk, it will have a great negative impact on the overall performance.

### Distributed Training

Similar to other modules of OneFlow, OneEmbedding supports distributed expansion natively. Users can refer to the README in [#dlrm](扩展阅读：DLRM) to start DLRM distributed training. You can also refer to [Global Tensor](../parallelism/03_consistent_tensor.md) for necessary prerequisites.

When using the OneEmbedding module for distributed expansion, please be careful:

- Currently, OneEmbedding only supports placement on all devices, and the parallelism must be the same as the world size. For example, when training with 4 cards in parallel, the parallelism of the Embedding table must be 4. It is not supported when the network is trained with 4 cards but the Embedding table parallelism is 2.
- The `persistent_path` parameter in the `store_options` configuration specifies the path of the storage. In parallel scenarios, it can be either a string representing a path or a `list`. If configured as a string representing a path, it represents the root directory under each rank in distributed parallelism. OneFlow will create a storage path based on the number of each rank under this root path, and the name format is `rank_id-num_rank`. If `persistent_path` is a `list`, rank will be configured individually for each item in the list.
- In parallel scenarios, the `capacity` in the `store_options` configuration represents the capacity of total Embedding table, but not the capacity of each rank. `cache_budget_mb` represents the video memory per GPU device.

## Extended Reading: DLRM

This article shows how to get started with OneEmbedding quickly.

Practical examples of OneEmbedding in DLRM tasks are prepared in the OneFlow model repository, please refer to https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/dlrm
