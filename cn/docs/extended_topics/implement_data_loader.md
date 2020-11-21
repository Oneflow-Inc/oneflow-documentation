# 自定义 DataLoader
如 [数据输入](../basics_topics/data_input.md) 一文所介绍，OneFlow 支持两种数据加载方式：直接使用 NumPy 数据或者使用 DataLoader 及其相关算子。

在大型工业场景下，数据加载容易成为训练的瓶颈。在其它的框架中，数据加载流水线往往作为单独的模块存在，需要针对不同场景进行调整，通用性不高。在 OneFlow 中，DataLoader 及相关预处理算子，与其它普通算子地位等同，可以享受与其它算子一样的流水加速效果，轻松解决大规模数据加载的痛点。

在 OneFlow 中使用 DataLoader，一般通过调用 `XXXReader` 加载文件中的数据，调用 `XXXDeocde` 等对数据进行解码或其它预处理，他们一起协同完成 Dataloader 的功能。

OneFlow 目前内置了一些文件格式的 [DataLoader](https://oneflow.readthedocs.io/en/master/search.html?q=reader&check_keywords=yes&area=default)。如果我们想使用 DataLoader 提高数据加载的效率，但是加载的数据格式暂时没有内置在 OneFlow 中，此时，我们可以自己实现 DataLoader，加载自定义的数据格式。

本文实现了一个 Mini Dataloader，在仓库中可查看[完整代码](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/data_loader/)。

作为示例，Mini Dataloader 支持的文件格式为：以逗号分隔的每行两列数字的纯文本文件(见代码中的 `part-000` 及 `part-001` 文件)：
```text
1.01,2.02
2.01,4.02
3.0,6.05
4.1,8.205
5,10
6.0,12.0
7.0,14.2
8.0,16.3
9.1,18.03
```

本文将以 Mini Dataloader 为例，对自定义格式的 DataLoader 的实现要点，进行讲解。

## Dataloader 的组成
完整的 Dataloader 一般包括两类 Op：

- Data Reader：负责将文件系统中的数据，加载到内存中的输入流，并最终将数据设置到 Op 的输出中。它又可以细分为 Loader 与 Parser 两部分，Loader 负责从文件系统中读取原始数据，Parser 负责将原始数据组织为 Data Reader Op 的输出
- Data Preprocessor：将 Data Reader Op 的输出的数据进行预处理，常见的预处理有图片解码、剪裁图片、解码等

对于一些简单的数据格式，不需要预处理，可以省略掉 Data Preprocessor，只使用 Data Reader 即可。

作为示例， Mini Dataloader 处理的数据格式虽然简单，但是依然实现了 DataReader 及 Data Preprocessor 两类 op，其中：

- `MiniReader` 负责从文件中读取数据，并按逗号分隔字符串，将文本转为浮点数据后，设置到 Op 的输出中，输出形状为每行两列
- `MiniDecoder` 负责将以上每行两列的输出进行分割，得到2个每行1列的输出 `x` 与 `y`

在 [test_mini_dataloader.py](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/data_loader/test_mini_dataloader.py) 中可以看到 Python 层次两者的使用：
```python
    miniRecord = MiniReader(
        "./",
        batch_size=batch_size,
        data_part_num=2,
        part_name_suffix_length=3,
        random_shuffle=True,
        shuffle_after_epoch=True,
    )

    x, y = MiniDecoder(
            miniRecord, name="d1"
        )
```

以下，我们将介绍 C++ 层次如何实现 Data Reader 算子与 Data Preprocessor 算子。

## Data Reader 算子
### Data Reader 的类关系
我们需要实现一个继承自 `DataReader` 的类，该类包含了两个重要对象 `loader_` 与 `parser_`，分别继承自 `Dataset` 与 `Parser`。

- `loader_` 的工作是从文件系统中读取数据至缓冲区，Op 作者通过重写 `Next` 方法编写这部分的逻辑
- `parser_` 的工作是将缓冲区中的数据，设置到 Op 的输出中，Op 作者通过重写 `Parser` 方法编写这部分的逻辑

当 Data Reader Op 工作时，会调用 `loader_` 中的相关方法打开文件系统中的文件，并调用 `loader_` 的 `Next` 方法按照 Op 作者预定的逻辑从文件系统读取数据，然后，再调用 `parser_` 的 `Parser` 方法，将数据设置到 Op 的输出中。

以下的伪代码展示了以上类关系和调用过程，实际代码比伪代码要复杂，并不是一一对应的关系：
```cpp
class DataReader{
    void Read(user_op::KernelComputeContext* ctx) {
    // 运行到此处，已经启动了多线程加速数据处理
    loader->next();
    parser_->Parse();
  }
    Dataset* loader_;
    Parser*  parser_;
};

class MiniDataReader : DataReader{
    loader_ = new MiniDataSet;
    parser_ = new MiniParser;
};

class MiniDataset: Dataset {
  MiniDataset() {
    //在文件系统中找到数据集，并打开文件，初始化输入流
    //...
  }

  Next() {
    // 从输入流中读取数据的逻辑
  }
};

class MiniParser: Parser {
  void Parse(){
    // 将 DataSet 中的数据 设置到 Op 的输出中
  }
};
```
在 Data Reader Op 的 Kernel 中，会触发 `DataReader` 的 `Read` 方法，进而完成以上伪代码所展示的一连串操作。

以下我们针对 MiniReader 算子的真实代码进行解析。

### Op 及 Kernel 注册
我们通过以下代码，注册了 MiniReader 的 Op：
```cpp
REGISTER_CPU_ONLY_USER_OP("MiniReader")
    .Output("out")
    .Attr<std::string>("data_dir")
    .Attr<std::int32_t>("data_part_num")
    .Attr<std::string>("part_name_prefix", "part-")
    .Attr<int32_t>("part_name_suffix_length", -1)
    .Attr<int32_t>("batch_size")
    .Attr<bool>("random_shuffle", false)
    .Attr<bool>("shuffle_after_epoch", false)
    .Attr<int64_t>("seed", -1)
    .Attr<int32_t>("shuffle_buffer_size", 1024)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      //...
      *out_tensor->mut_shape() = Shape({local_batch_size, 2});
      *out_tensor->mut_data_type() = DataType::kDouble;
      //...
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      //...
    });
```
可以看到，因为 Data Reader 是比较特殊的 Op，只有输出，没有输入（数据来自文件系统，而不是神经网络中的某个上游节点），因此我们只通过 `Out` 方法设置了输出，并在 `SetTensorDescInferFn` 设置了输出的性质为每行2列，数据类为 `DataType::kDouble`。同理，在设置 `SetGetSbpFn` 中设置 SBP Signature 时，只需要设置输出的 SBP 属性，我们将其设置为 Split(0)。

而设置的各种属性（`data_dir`、`data_part_num` 等），沿用了 [OFRecord 数据集](../extended_topics/ofrecord.md#ofrecord) 中关于文件命名规范的要求，这使得我们可以复用 OneFlow 中已有的相关代码，像 [加载 OFRecord 数据集](../extended_topics/how_to_make_ofdataset.md#ofrecord_1) 那样，加载我们自定义格式的文件。

接着看这个 Op 的 Kernel 实现：
```cpp
class MiniReaderKernel final : public user_op::OpKernel {
 public:
  //...

  std::shared_ptr<user_op::OpKernelState>
  CreateOpKernelState(user_op::KernelInitContext* ctx) override{
    std::shared_ptr<MiniReaderWrapper> reader(new MiniReaderWrapper(ctx));
    return reader;
  }

  void Compute(user_op::KernelComputeContext* ctx,
               user_op::OpKernelState* state) override {
    auto* reader = dynamic_cast<MiniReaderWrapper*>(state);
    reader->Read(ctx);
  }
  //...
};

REGISTER_USER_KERNEL("MiniReader")
    .SetCreateFn<MiniReaderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("out", 0) == DataType::kDouble));
```
依据 [自定义 Op](../extended_topics/user_op.md) 一文中的知识，我们知道 `MiniReaderKernel::Compute` 负责 Op 的运算逻辑。不过此处使用使用的 `Compute` 是包含2个参数的重载，我们有必要介绍下它的第二个参数 `OpKernelState`。

当我们进行 `Compute` 时，有时除了从 `ctx` 获取的信息外，我们还需要维护一些其他的对象，这种对象不需要反复创建，但是其中的信息状态可能随着 `Compute` 多次调用而改变。为了应对这种需求，OneFlow 提供了2个参数的 `Compute` 的重载，为了使用它，我们必须同时重写 `CreateOpKernelState`，`CreateOpKernelState` 的作用是返回一个 `user_op::OpKernelState` 派生类对象，这个对象，将在 `Compute` 调用时，作为第二个参数传递。

为此，我们只需要将除 `ctx` 外想要维护的信息，封装为 `user_op::OpKernelState` 的派生类，并在 `CreateOpKernelState` 实例化并返回即可。

具体到我们实现的 Mini Reader 的 Kernel，我们先实现了一个继承自 `user_op::OpKernelState` 的类 `MiniReaderWrapper`， 它是对 `MiniDataReader` 的简单封装，之所以封装一层 `MiniReaderWrapper` 而不直接使用 `MiniDataReader`，仅仅是为了符合以上所述的 OneFlow 框架要求。
```cpp
class MiniReaderWrapper final : public user_op::OpKernelState {
 public:
  explicit MiniReaderWrapper(user_op::KernelInitContext* ctx) : reader_(ctx) {}
  ~MiniReaderWrapper() = default;

  void Read(user_op::KernelComputeContext* ctx) { reader_.Read(ctx); }

 private:
  data::MiniDataReader reader_;
};
```

然后，重写 `CreateOpKernelState`，在其内部创建一个 `MiniReaderWrapper` 对象：
```cpp
  std::shared_ptr<user_op::OpKernelState>
  CreateOpKernelState(user_op::KernelInitContext* ctx) override{
    std::shared_ptr<MiniReaderWrapper> reader(new MiniReaderWrapper(ctx));
    return reader;
  }
```

这样，在适当的时机，OneFlow 就会自动调用 `CreateOpKernelState` 创建对象，并将其作为第2个参数传递给 `Compute`。我们可以在 `Compute` 中拿到这个对象，并使用：
```cpp
    auto* reader = dynamic_cast<MiniReaderWrapper*>(state);
    reader->Read(ctx);
```
可以看到，在 MiniReader 的 Kernel 中，我们仅仅是简单调用了 `MiniReaderWrapper::Reader`，这会触发上文伪代码中所提及的 `DataReader::Read` 流程。

### MiniDataReader
上文伪代码中已经提及，在 `MiniDataReader` 内部，会实例化一个 `MiniDataset` 并赋值给 `loader_` 指针。
以下是真实代码：
```cpp
class MiniDataReader final : public DataReader<TensorBuffer> {
 public:
  MiniDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new MiniDataset(ctx));
    parser_.reset(new MiniParser());
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
    }
    int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
};
```
可以看到，除了我们自己继承自 `DataSet` 的 `MiniDataset` 类之外，OneFlow 还内置了其他的 `XXXDataSet`，它们可以在已有的 `DataSet` 基础上增加额外功能，如 `RandomShuffleDataset` 用于 shuffle，`BatchDataset` 用于批量读取数据。
一切完成后，最后调用 `StartLoadThread`，顾名思义，启动加载线程，在 `StartLoadThread` 中，最终会触发重写的 `MiniDataset::Next` 方法。

以上 `MiniDataReader` 的构造，可以作为模板，没有特殊要求，在实现自定义的 DataLoader 过程中，不需要修改。


### MiniDataset
对于 `MiniDataSet`，我们只需要关心它的构造函数以及重写的 `Next` 方法。

构造函数主要是通过 `Attr` 获取用户的配置，然后根据用户配置，初始化输入流。以下代码中的 `JoinDirPath` 内部，主要根据数据集文件名的约定（前缀、文件数目，文件名编号是否补齐等），获取所有的文件名称；而 `InitInStream` 是将数据集中的文件，都初始化为 OneFlow 封装的输入流（`in_stream_` 成员），这在后续的 `Next` 方法中会使用。
```cpp
  MiniDataset(user_op::KernelInitContext* ctx) {
    current_epoch_ = 0;
    shuffle_after_epoch_ = ctx->Attr<bool>("shuffle_after_epoch");

    //Join Dir Path
    JoinDirPath(ctx);

    // in stream
    InitInStream(ctx);
  }
```

从文件中加载的逻辑，写在 `Next` 虚函数中：
```cpp
  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    LoadTargetPtr sample_ptr(new TensorBuffer());

    std::string sampleline;
    if (in_stream_->ReadLine(&sampleline) != 0) {
      ShuffleAfterEpoch();
      in_stream_->ReadLine(&sampleline);
    }

    auto numbers = CommaSplit(sampleline);
    sample_ptr->Resize(Shape({2}), DataType::kDouble);
    auto pNums = sample_ptr->mut_data<double>();
    pNums[0] = std::stod(numbers[0]);
    pNums[1] = std::stod(numbers[1]);
    ret.push_back(std::move(sample_ptr));

    return ret;
  }
```
在以上代码中，我们通过调用 `in_stream_` 的 `ReadLine` 方法，将文件中的数据，读取至 `string` 对象 `sampleline` 中。然后通过 `CommaSplit` 等操作，将字符串按逗号分隔，并转为浮点数，放置到 `TensorBuffer` 对象中。

值得一提的是，`in_stream_` 有2种方法从文件中读取数据，分别是：
```cpp
int32_t PersistentInStream::ReadLine(std::string* l);
int32_t PersistentInStream::ReadFully(char* s, size_t n);
```
`ReadLine` 读取文件中的一行，至 `l` 对象；`ReadFully` 读取 `n` 个字节的数据，至 `s` 所指向的内存。均以0作为成功时的返回值。

`MiniDataSet` 完成从文件到内存缓冲区的工作，接着，我们将使用 `MiniParser`，将缓冲区中的内容，设置到 Op 的输出中。

### MiniParser
`MiniParser` 继承自 `Parser`，我们只需要重写其中的 `Parser` 方法即可：
```cpp
class MiniParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;

  void Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
             user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    double* dptr = out_tensor->mut_dptr<double>();

    MultiThreadLoop(batch_data->size(), [&](size_t i) {
      TensorBuffer* buffer = batch_data->at(i).get();
      dptr[i*2]= *(buffer->data<double>());
      dptr[i*2+1]= *(buffer->data<double>()+1);
    });
  }
};
```
`Parser` 包含2个参数，其中 `batch_data` 其实是一个封装了的 `vecotr`，这个容器内的每个元素，就是之前 `MiniDataSet` 通过 `Next` 读取的数据。 参数 `ctx` 使得我们可以获取 Op 的信息，在这里，我们主要通过 `ctx` 获取输出，并获取指向输出缓冲区的指针 `dptr`。

注意，我们将 `batch_data` 中的数据设置到 Op 的输出 `dptr` 的过程中，使用了宏 `MultiThreadLoop`。`MultiThreadLoop` 可以让我们的循环逻辑在多线程中执行，它接受2个参数，第一个参数为循环的总次数；第二个参数是一个回调函数，原型为 `void callback(size_t i)`，OneFlow 会创建多个线程，然后并发调用这个回调函数。回调函数的参数 `i` 表明了当前循环的序号，使得我们可以根据 `i` 来划分数据，完成自己的业务逻辑。

在以上的代码中，我们通过 `batch_data->at(i).get()` 获取了缓冲区的第 `i` 个的数据，然后将其设置到输出的内存区的第 `i` 行的位置，一共2列。

## Data Preprocessor 算子
Data Preprocessor 算子，其实就是一种普通的算子，它接受 `DataReader` 的输出作为自己的输入，然后通过运算后，输出一个或者多个 Blob。

在 [ofrecord_decoder_ops.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/ops/ofrecord_decoder_ops.cpp) 可以看到针对 OFRecord 数据的各种预处理操作（以解码为主）。

我们的 Mini Dataloader 处理的数据比较简单，因此 MiniDecoder 所做的工作也很简单，仅仅是将 `DataReader` 所输出的每行2列的数据，拆分为2个每行1列的输出 `x` 与 `y`。

Mini Decoder 的 Op 注册为：
```cpp
REGISTER_CPU_ONLY_USER_OP("mini_decoder")
    .Input("in")
    .Output("x")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* out_tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      // 设置输入输出 Blob 的属性
      // ...
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      //...
    });
```

Mini Decoder 的 Kernel 的实现：
```cpp
class MiniDecoderKernel final : public user_op::OpKernel {
  //...
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_blob_y = ctx->Tensor4ArgNameAndIndex("y", 0);

    int64_t record_num = in_blob->shape().At(0);

    const double* input = in_blob->dptr<double>();
    double* out_dptr_x = out_blob_x->mut_dptr<double>();
    double* out_dptr_y = out_blob_y->mut_dptr<double>();

    MultiThreadLoop(record_num, [&](size_t i){
      *(out_dptr_x + i) = *(input+i*2);
      *(out_dptr_y + i) = *(input+i*2 + 1);
    });

  }
  //...
};
```
可见，在 `MiniDecoderKernel::Compute` 中主要是获取到输入 `in_blob`， 然后在多线程循环 `MultiThreadLoop` 中，将输入的数据拆分到 `out_dptr_x` 与 `out_dptr_y` 中，它们分别对应了输出 `x` 与 `y`。



## 自定义 DataLoader 的使用
如 [自定义 Op](./user_op.md) 一文中所描述，要使用 C++ 层编写的 Op，还需要在 Python 层封装一个 Python Wrapper。这些工作放到了 [test_mini_dataloader.py](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/data_loader/test_mini_dataloader.py)中：
```python
def MiniDecoder(
    input_blob,
    name = None,
):
    if name is None:
        name = "Mini_Decoder_uniqueID"
    return (
        flow.user_op_builder(name)
        .Op("mini_decoder")
        .Input("in", [input_blob])
        .Output("x")
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

def MiniReader(
    minidata_dir: str,
    batch_size: int = 1,
    data_part_num: int = 2,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    random_shuffle: bool = False,
    shuffle_after_epoch: bool = False,
    shuffle_buffer_size: int = 1024,
    name = None,
):
    if name is None:
        name = "Mini_Reader_uniqueID"

    return (
        flow.user_op_builder(name)
        .Op("MiniReader")
        .Output("out")
        .Attr("data_dir", minidata_dir)
        .Attr("data_part_num", data_part_num)
        .Attr("batch_size", batch_size)
        .Attr("part_name_prefix", part_name_prefix)
        .Attr("random_shuffle", random_shuffle)
        .Attr("shuffle_after_epoch", shuffle_after_epoch)
        .Attr("part_name_suffix_length", part_name_suffix_length)
        .Attr("shuffle_buffer_size", shuffle_buffer_size)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
```

在 `test_mini_dataloader.py` 中，我们使用自己实现的 `MiniReader` 与 `MiniDecoder` 加载并解码了数据集（`part-000` 与 `part-001`）中的数据，完成了一次训练。

## Mini Dataloader 的编译与测试
进入到本文对应的 [data_loader](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/data_loader/) 目录。
修改 `Makefile` 文件中的 `ONEFLOW_ROOT` 变量为 OneFlow 源码路径。
然后通过

```bash
make
```
可生成 `miniloader.so` 文件。

然后运行 `test_mini_dataloader.py` 脚本，可以使用 Mini Dataloader 加载数据并完成训练。
```bash
python test_mini_dataloader.py
```
