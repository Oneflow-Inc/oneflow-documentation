# 模型训练可视化
Oneflow支持将训练生成的中间结果以日志文件的形式保存到本地，可视化后端通过实时读取日志文件，将训练过程产生的数据实时展示到可视化前端。

目前，Oneflow支持的可视化类型分为以下几种：

| 可视化类型 | 描述                     |
| ---------- | ------------------------ |
| 模型结构   | 结构图、计算图(后续支持) |
| 标量数据   | 标量数据                 |
| 媒体数据   | 文本、图像               |
| 统计分析   | 数据直方图、数据分布图   |
| 降维分析   | 数据降维                 |
| 超参分析   | 超参数                   |
| 异常检测   | 异常数据检测             |

本文将介绍
* 如何初始化summary日志文件

* 如何生成结构图日志

* 如何生成标量数据日志

* 如何生成媒体数据日志

* 如何生成统计分析数据日志

* 如何生成降维分析日志

* 如何生成超参分析日志

* 如何生成异常检测日志

本文中提到的可视化日志具体使用方式可参考test_summary.py 文件，

具体可视化效果参考[之江天枢人工智能开源平台](http://tianshu.org.cn/?/course)用户手册可视化部分。

## 初始化

首先定义一个用于存放日志文件的目录 logdir, 我们定义以下计算函数调用 flow.summary.create_summary_writer接口创建日志文件写入方法。
```python
@flow.global_function(function_config=func_config)
def CreateWriter():
    flow.summary.create_summary_writer(logdir)
```
再调用CreateWriter()这个函数，就可以完成summary writer对象的创建啦



## 生成结构图日志

首先通过调用flow.summary.Graph接口生成graph对象，然后再通过调用graph.write_structure_graph方法将graph写入到日志文件

```python
graph = flow.summary.Graph(logdir)
graph.write_structure_graph()
```



## 生成标量数据日志

通过定义flow.summary.scalar接口来创建标量数据写入方法

```python
@flow.global_function(function_config=func_config)
def ScalarJob(
    value: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.float),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((1000,), dtype=flow.int8),
):
    flow.summary.scalar(value, step, tag)
```

再调用ScalarJob()这个函数，就可以将标量数据写入日志文件

```python
value = np.array([1], dtype=np.float32)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("scalar", dtype=np.int8)
ScalarJob([value], [step], [tag])
```



## 生成媒体数据日志

####  文本数据日志生成

通过定义flow.summary.pb接口来创建文本数据写入方法

```python
@flow.global_function(function_config=func_config)
def PbJob(
    value: flow.typing.ListNumpy.Placeholder((1500,), dtype=flow.int8),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
):
    flow.summary.pb(value, step=step)
```

使用的时候，首先我们要定义一个字符串列表，然后通过调用flow.summary.text接口生成protobuf message，再将其转化成字符串，最后调用PbJob()函数将文本数据写入日志文件

```python
net = ["vgg16", "resnet50", "mask-rcnn", "yolov3"]
pb = flow.summary.text(net)
value = np.fromstring(str(pb), dtype=np.int8)
step = np.array([i], dtype=np.int64)
PbJob([value], [step])
```

#### 图像数据日志生成

通过定义flow.summary.image接口来创建图像数据写入方法

```python
@flow.global_function(function_config=func_config)
def ImageJob(
    value: flow.typing.ListNumpy.Placeholder(
    shape=(100, 2000, 2000, 4), dtype=flow.uint8
),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((10,), dtype=flow.int8),
):
    flow.summary.image(value, step=step, tag=tag)
```



```python
images = _read_images_by_cv(image_files)
images = np.array(images, dtype=np.uint8)
imageRed = np.ones([512, 512, 3]).astype(np.uint8)
Red = np.array([0, 255, 255], dtype=np.uint8)
imageNew = np.multiply(imageRed, Red)
mageNew = np.expand_dims(imageNew, axis=0)
images = np.concatenate((images, imageNew), axis=0)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("image", dtype=np.int8)
ImageJob([images], [step], [tag])
```



## 生成统计分析数据日志

通过定义flow.summary.histogram接口来创建统计分析数据写入方法

```python
@flow.global_function(function_config=func_config)
def HistogramJob(
    value: flow.typing.ListNumpy.Placeholder((200, 200, 200), dtype=flow.float),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((9,), dtype=flow.int8),
):
    flow.summary.histogram(value, step, tag)
```

我们通过传入value, step 和 tag，调用HistogramJob()函数即可将统计分析数据写入日志文件

```python
value = np.random.rand(100, 100, 100).astype(np.float32)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("histogram", dtype=np.int8)
HistogramJob([value], [step], [tag])
```



## 生成超参分析日志

首先，创建一个包含超参数据的map harams，然后通过flow.summary.hparams函数生成protobuf message，转化成字符串之后调用PbJob()函数将超参分析数据写入日志文件

```python
hparams = {
    flow.summary.HParam("learning_rate", flow.summary.RealRange(1e-2, 1e-1)): 0.02,
    flow.summary.HParam("dense_layers", flow.summary.IntegerRange(2, 7)): 5,
    flow.summary.HParam(
         "optimizer", flow.summary.ValueSet(["adam", "sgd"])
    ): "adam",
    flow.summary.HParam("accuracy", flow.summary.RealRange(1e-2, 1e-1)): 0.001,
    flow.summary.HParam("magic", flow.summary.ValueSet([False, True])): True,
    flow.summary.Metric("loss", float): 0.02,
    "dropout": 0.6,
}

pb2 = flow.summary.hparams(hparams)
value = np.fromstring(str(pb2), dtype=np.int8)
step = np.array([i], dtype=np.int64)
PbJob([value], [step])
```



## 生成降维分析日志

首先通过调用flow.summary.Projector()接口生成projector对象，然后通过projecotr.create_embedding_projector()函数创建embedding_projector对象，最后通过调用projecotr.embedding_projector()函数将降维分析日志写入日志文件

```python
projecotr = flow.summary.Projector(logdir)
projecotr.create_embedding_projector()

value_ = np.random.rand(10, 10, 10).astype(np.float32)
label = (np.random.rand(10) * 10).astype(np.int64)
x = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
sample_name = "sample"
sample_type = "image"
step = 1
tag_embedding = "embedding_projector"
projecotr.embedding_projector(
    value=value,
    label=label,
    tag=tag_embedding,
    step=step,
    sample_name=sample_name,
    sample_type=sample_type,
    x=x,
)
```



## 生成异常检测日志

首先通过调用flow.summary.Projector()接口生成projector对象，然后通过projecotr.create_exception_projector()函数创建exception_projector对象，最后通过调用projecotr.exception_projector()函数将异常检测日志写入日志文件

```python
projecotr = flow.summary.Projector(logdir)
projecotr.create_exception_projector()

value_ = np.random.rand(10, 10, 10).astype(np.float32)
x = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
sample_name = "sample"
sample_type = "image"
step = 1
tag_embedding = "exception_projector"
projecotr.exception_projector(
    value=value,
    tag=tag_exception,
    step=step,
    sample_name=sample_name,
    sample_type=sample_type,
    x=x,
)
```

