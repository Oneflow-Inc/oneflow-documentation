# Large-Scale Embedding Solution: OneEmbedding

Embedding has become the foundamental opeartion in recommender systems, and it has also spread to many fields outside recommender systems. Each framework provides basic operators for embedding, for example, you can use `flow.nn.Embedding` in OneFlow:

```python
import numpy as np
import oneflow as flow
indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
embed = flow.nn.Embedding(10, 3)
y = embed(indices)
```

OneEmbedding is the large-scale Embedding solution that OneFlow provides to solve the problem of large-scale deep recommender systems. OneEmbedding has the following advantages compared to ordionary opeartors:

1. With Flexible hierarchical storage, OneEmbedding can place the Embedding table on GPU memory, CPU memory, or SSD, and uses high-speed devices as caches for low-speed devices to achieve both speed and capacity.

2. OneEmbedding supports dynamic expansion.

## Get Start to OneEmbedding Quickly

The following steps is an example of getting started with OneEmbeeding quickly: 

- Creat Embedding table with `make_table`
- Configure the storage attribute of the Embedding table
- Instantiate Embedding
- Construct Graph for training

### Creat Embedding Table with `make_table`

By importing relevant package and the following codes, you can creat Embedding table with `make_table`.OneEmbedding supports simultaneous creation of multiple Embedding table. The following codes creat three Embedding table, with an initial feature length of `128`, and the size (rows) of each table are: `39884407`, `39043` and `17289`.

```python
import oneflow as flow
import numpy as np
import oneflow.nn as nn

table_size_array = [39884407, 39043, 17289]
vocab_size = sum(table_size_array)
num_tables = len(table_size_array)
embedding_size = 128    
scales = np.sqrt(1 / np.array(table_size_array))

tables = [
    flow.one_embedding.make_table(
        flow.one_embedding.make_uniform_initializer(low=-scale, height=scale)
    )
    for scale in scales
]
```

You can specify the initialization method when create Embedding table with `make_table`. The above codes initialize Embedding table with the method `uniform`. The created Embedding table will be stored in variable `tables`. 

Click [make_table]() and [make_uniform_initializer]() to check more detailed description.

### Configure the Storage Attribute of the Embedding Table

Then run the following codes to configure the storage attribute of the Embedding table:

```python
store_options = flow.one_embedding.make_cached_ssd_store_options(
    cache_budget_mb=8142,
    persistent_path="/your_path_to_ssd", 
    capacity=vocab_size,
    size_factor=1,   			
    physical_block_size=512
)
```

By calling `make_cached_ssd_store_options` here, you can store Embedding table on SSD and use GPU as cache. For the meaning of specific parameters, please refer to [make_cached_ssd_store_options API 文档]().

In addition, you can use pure GPU as storage; or use CPU memory to store Embedding table, but use GPU as cache. For more details, please refer to [make_cached_ssd_store_options]() and [make_cached_host_mem_store_option]().

### Instantiate Embedding

After the above configuration is completed, you can use `MultiTableEmbedding` to get the instantiated Embedding layer.

```python
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

Among them, `tables` is the Embedding table previously constructed by `make_table`, `store_options` is the previously configured storage attribute, `embedding_dim` is the feature dimension, `dtype` is the data type of the feature vector, `key_type` is the data type of feature ID.

For more detailes, please refer to [one_embedding.MultiTableEmbedding]().

### Construct Graph for Training

Currently, OneEmbedding is only supported in Graph mode. 

In the following example, we construct a simple Graph class that includes `embedding` and `mlp` layers.

```python
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

### Storage and Loading

You can store Embedding table with the method of `MultiTableEmbedding.save_snapshot`:

```python
embedding.save_snapshot('./my_snapshot')
```

If you want to load a Embedding table from a stored snapshot, you should call `MultiTableEmbedding.load_snapshot`.

For more detailed information, please refer to [MultiTableEmbedding.save_snapshot]() and [MultiTableEmbedding.load_snapshot]().

## Dynamically insert New Feature ID

It is important to note that OneEmbedding supports dynamic insertion of new feature ID. As long as the storage medium has sufficient capacity, there is no upper limit on the number of feature IDs. Specifically, when querying, the feature ID can go beyond the scope when the Embedding table was created.

For example, in the above codes, the Embedding table was cerated with the size of `39940739` （`vocab_size = sum(table_size_array)`), but during training, it is legal even if the `ids` in `embedding = self.embedding_lookup(ids)` exceeds `39940739`.
  
## Extended Reading: DLRM

This article shows how to get started with OneEmbedding quickly.

Practical examples of OneEmbedding in DLRM tasks are prepared in the OneFlow model repository, please refer to https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/dlrm
