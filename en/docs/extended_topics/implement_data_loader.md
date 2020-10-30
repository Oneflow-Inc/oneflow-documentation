# Customize DataLoader

As described in [Data Input](../basics_topics/data_input.md), OneFlow supports two ways to load data: one is directly use Numpy data, the other one is use DataLoader and some relative operators. 

Under the large industrial scene, Data Loading can easily become the bottleneck through the training process. Since we use DataLoader and some preprocessing operators, OneFlow's acceleration mechanism helps to load and preprocess data more efficiently, which can solve the problem. 

To use DataLoader in OneFlow, we usually apply `XXXReader` to load the file data, and we use `XXXDecode` to decode or preprocess the data. These two operators work together to complete the function of DataLoader.  

Now OneFlow has built some data format's [DataLoader](https://oneflow.readthedocs.io/en/master/search.html?q=reader&check_keywords=yes&area=default) internally. If we want to use DataLoader to promote the efficiency of data loading, however, the DataLoader for the corresponding data format is not yet built in OneFlow. At this time, we can implement our own DataLoader to load the customized data format.  

In this article we implement a Mini Dataloader. You can check the [Complete Code](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/en/docs/code/extended_topics/data_loader/) in this repository. 

As an example, the data format that Mini Dataloader supported is : A plain text file with two columns of numbers separated by commas (See the `part-000` and `part-001` file in code): 

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

This article will take Mini Dataloader as an example to explain the key points of implementing customized DataLoader. 

# The composition of Dataloader

A complete Dataloader generally includes two types of Op: 

- Data Reader: Which is responsible for loading the data in file system to the input stream of memory and setting the data to the Op's output. 
- Data Decoder: The Data Decoder decodes and output the data in Data Reader Op. 

For some simple data formats, which is no need for decoding. We can omit the Data Decoder and just use Data Reader. 

As an example, since the data format processed by Mini Dataloader is simple. We still implement the Data Reader and Data Decoder. Among these two Ops: 

- `MiniReader` is responsible for reading data from files and split strings by commas. Convert the text to the float and set to the Op's output. The output shape is two columns of each row. 
- `MiniDecoder` is responsible for splitting the two columns of each row output in above and get two outputs `x` and `y`, both of them shape is one column of each row. 

In [test_mini_dataloader.py](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/en/docs/code/extended_topics/data_loader/test_mini_dataloader.py) we can see the use of both Ops at Python level:

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

We will introduce how to implement Data Reader and Data Decoder in C++ backend below. 

# Data Reader operator

## The class relationship in Data Reader

We need to implement a class that inherits from `DataReader`, this class includes two important objects `loader_` and `parser_`, which inherits from `Dataset` and `Parser` separately. 

- `loader_` 's job is to read data from the file system to buffer. The Op's author build the logic by overriding the `Next` function. 
- `parser_` 's job is to set the data in buffer to Op's output. The Op's author build the logic by overriding the `Parser` function. 

When Data Reader Op works, it will call the relative function in `loader_` to open files in the file system, and then call the `Next` function in `loader_` to read data from the file system according to the logic built by Op's author. 

The pseudocode below shows the class relationship and the calling procedure. The actual code is more complicated than the pseudocode and it's not a one-to-one relationship: 

```c++
class DataReader{
    void Read(user_op::KernelComputeContext* ctx) {
    // OneFlow already starts multi-threads to accelerate data processing when code runs here. 
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
    // Find and open dataset in the file system, initialize the input stream. 
    //...
  }

  Next() {
    // The logic of reading data from input stream. 
  }
};

class MiniParser: Parser {
  void Parse(){
    // Set the data from DataSet to Op's output. 
  }
};
```

In Data Reader Op's Kernel, it will trigger the `Read` function in `DataReader` and complete the sequence of operations which is shown in the pseudocode above. 

## The registration of Op and Kernel

We register the MiniReader's Op through the code below: 

```c++
REGISTER_CPU_ONLY_USER_OP("MiniReader")
    .Output("out")
    .Attr<std::string>("data_dir")
    .Attr<std::int32_t>("data_part_num")
    .Attr<std::string>("part_name_prefix", std::string("part-"))
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

As we can see, because Data Reader is a special Op, it has only output, no input (data comes from the file system, instead of some upstream nodes), we only use `Out` function to set the output, and set the output shape as two columns per row in `SetTensorDescInferFn`, the data type is `DataType::kDouble`. In the same way, when we set SBP Signature in `SetGetSbpFn`, we only need to set output's SBP attribution. In this case, we set it as Split(0). 

The other attributions (like `data_dir`, `data_part_num`, etc.) follow the requirement of file naming conventions in [The OFRecord Data Format](../extended_topics/ofrecord.md#ofrecord). It allows us to reuse some related code in OneFlow to load customized data format like [The method to load OFRecord dataset](../extended_topics/how_to_make_ofdataset.md#ofrecord_1). 

Then let's look at the implementation of Op's Kernel: 

```c++
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

According to the knowledge in [Customize Op](../extended_topics/user_op.md), we have known that `MiniReaderKernel::Compute` function is responsible for the Op's compute logic. However, we use an override version of `Compute` that includes two parameters here. It's necessary to introduce the second parameter `OpKernelState`. 

When we call the `Compute`, we need to maintain other objects in addition to get information from `ctx`. This type of object does not need to be created repeatedly, but their state of information may change as `Compute` function is called multiple times. In response to this need, OneFlow provides a override version with two parameters of `Compute`. In order to use it, we need to override `CreateOpKernelState` at the same time. `CreateOpKernelState` returns a `user_op::OpKernelState` derived class object. This object will be the second parameter to be transmitted when `Compute` is called. 

So we only need to encapsulate the information we want to maintain, in addition to the `ctx`, as a derived class of `user_op::OpKernelState`, instantiate and return it in `CreateOpKernelState`. 

In our Mini Reader's Kernel, we first implement a class `MiniReaderWarapper` that is inherited from `user_op::OpKernelState`.  It is a simple encapsulation of `MiniDataReader`, the reason why we encapsulate `MiniReaderWrapper` instead of using `MiniDataReader` directly is that to meet the requirements of OneFlow. 

```c++
class MiniReaderWrapper final : public user_op::OpKernelState {
 public:
  explicit MiniReaderWrapper(user_op::KernelInitContext* ctx) : reader_(ctx) {}
  ~MiniReaderWrapper() = default;

  void Read(user_op::KernelComputeContext* ctx) { reader_.Read(ctx); }

 private:
  data::MiniDataReader reader_;
};
```

Then, we override `CreateOpKernelState`, create a `MiniReaderwrapper` object internally. 

```c++
  std::shared_ptr<user_op::OpKernelState>
  CreateOpKernelState(user_op::KernelInitContext* ctx) override{
    std::shared_ptr<MiniReaderWrapper> reader(new MiniReaderWrapper(ctx));
    return reader;
  }
```

In this way, OneFlow will call `CreateOpKernelState` function to create object automatically in appropriate time and transmit it to `Compute` as the second parameter. We can get this object in `Compute`, and use: 

```c++
    auto* reader = dynamic_cast<MiniReaderWrapper*>(state);
    reader->Read(ctx);
```

As we can see, In MiniReader's Kernel, we just simply call `MiniReaderWrapper::Reader`, it will trigger the procedure of `DataReader::Read` that is mentioned in above pseudocode. 

## MiniDataReader

As we mentioned in above pseudocode. In `MiniDataReader`, it will instantiate a `MiniDataset` and assign to the `loader_` pointer. 

Here is the code: 

```c++
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

In addition to inheriting our `Dataset`'s `MiniDataset` class, OneFlow also build other `XXXDataset`, they can add additional features in the base of existing `Dataset`. For example, `RandomShuffleDataset` can be used to shuffle data, `BatchDataset` can be used to read batch data. When it is all done, we finally call `StartLoadThread`, which is used to start the loading thread. We will trigger the override function `MiniDataset::Next` in `StartLoadThread`. 

The above construction of `MiniDataReader` can be used as a template. If you have no special requirements, you don't need to modify it in custom DataLoader. 

## MiniDataset 

For `MiniDataSet`, we only need to focus on the constructor and overridden `Next` function. 

The constructor obtains user's settings through `Attr`. Then it will initialize the input stream according to the user's settings. In the following code, `JoinDirPath` is used to obtain all filenames according to the convention of dataset (the prefix, the amount of files, whether the filename number is complete, etc.). And `InitInStream` is to initialize the file in dataset as input stream (The member of `in_stream`), which is encapsulated by OneFlow, it will be used in `Next` function later. 

```c++
  MiniDataset(user_op::KernelInitContext* ctx) {
    current_epoch_ = 0;
    shuffle_after_epoch_ = ctx->Attr<bool>("shuffle_after_epoch");

    //Join Dir Path
    JoinDirPath(ctx);

    // in stream
    InitInStream(ctx);
  }
```

The logic of loading from files is written in virtual function `Next`: 

```c++
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

In the above code, we call `in_stream_`'s `ReadLine` function to load file data into `string` object `sampleline`. Then we call`CommaSplit` to split the string by commas, convert to float, and place it to `TensorBuffer` object. 

It is worth to mentioning that `_in_stream_` has two ways to read data from files: 

```c++
int32_t PersistentInStream::ReadLine(std::string* l);
int32_t PersistentInStream::ReadFully(char* s, size_t n);
```

`ReadLine` function read a row of file to `l` object; `ReadFully` function read `n` bytes data to the memory pointed by `s`. Both functions use 0 as return value on success. 

`MiniDataset` complete the process of file to memory buffer. In next section, we will use `MiniParser` to set the buffer's content to Op's output. 

## MiniParser

`MiniPaser` inherits from `Parser`, we just need to override the `Parser` function. 

```c++
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

`Parser` includes 2 parameters, where `batch_data` is an encapsulated `vector`. Each element in this container is the data previously read by `MiniDataSet` through `Next` function. The parameter `ctx` enables us to get the information of Op. Here we mainly obtain the output through `ctx` and get the pointer `dptr` to the output buffer. 

Notice that we use macro `MultiThreadLoop` in the procedure of setting `batch_data`'s data to the Op's output `dptr`. `MultiThreadLoop` allows our loop logic to be executed in multiple threads. It accept 2 parameters, the first parameter is the total number of loop; the second parameter is a callback function, its prototype is `void callback(size_t, i)`. OneFlow will create multiple threads, and call the callback function concurrently. The callback function's parameter `i` indicates the serial number of current loop, allowing us to divide data according to `i` and complete the business logic. 

In the above code, we get the `i`th data in buffer through `batch_data->at(i).get()`, and set it to the location of the `i`th row in the output memory area. There are two columns in total. 

## Data Decoder Operator

Data Decoder operator is a normal operator. It accept the output of `DataReader` as its input, output one or multiple Blobs after some operations. 

In  [ofrecord_decoder_ops.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/ops/ofrecord_decoder_ops.cpp), we can see various decoders for OFRecord data. 

The data processed by our Mini Dataloader is simple, so what MiniDecoder does is very simple. It just splits two columns data output from `DataReader` into two one-column outputs as `x` and `y`. 

Mini Decoder's Op is registered as: 

```c++
REGISTER_CPU_ONLY_USER_OP("mini_decoder")
    .Input("in")
    .Output("x")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* out_tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      // Set the input, output Blob's attribution 
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

The implementation of Mini Decoder's Kernel is as follow: 

```c++
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

We mainly get the input `in_blob` in `MiniDecoderKernel::Compute`, and then in multiple threads loop `MultiThreadLoop`, split the input data into `out_dptr_x` and `out_dptr_y`, they correspond to the output `x` and `y`. 



# The use of customized DataLoader

As  described in [customized user op](./user_op.md), if we want to use the Op built in C++ backend, we need to encapsulate a Python Wrapper in Python level. Some related work is put in [test_mini_dataloader.py](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/data_loader/test_mini_dataloader.py): 

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

In `test_mini_dataloader.py`, we use our implemented `MiniReader` and `MiniDecoder` to load and decode the data in dataset (`part-000` and `part-001`), complete a epoch of training. 

# Compile and test Mini Dataloader

Check into the corresponding directory  [data_loader](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/data_loader/) for this article. 

Change `Makefile`'s `ONEFLOW_ROOT` variable as the directory of OneFlow's source code. 

And then use

```bash
make
```

to generate `miniloader.so` file. 

Run `test_mini_dataloader.py` script, then we can use Mini Dataloader to load data and complete training. 

```bash
python test_mini_dataloader.py
```



