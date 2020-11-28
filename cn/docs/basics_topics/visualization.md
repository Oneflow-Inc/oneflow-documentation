# 模型训练可视化
OneFlow 支持将训练生成的中间结果以日志文件的形式保存到本地，可视化后端通过实时读取日志文件，将训练过程产生的数据实时展示到可视化前端。

目前，OneFlow 支持的可视化类型分为以下几种：

| 可视化类型 | 描述                     |
| ---------- | ------------------------ |
| 模型结构   | 结构图、计算图(后续支持) |
| 标量数据   | 标量数据                 |
| 媒体数据   | 文本、图像               |
| 统计分析   | 数据直方图、数据分布图   |
| 超参分析   | 超参数                   |
| 降维分析   | 数据降维                 |
| 异常检测   | 异常数据检测             |

本文将介绍：

- 如何初始化 summary 日志文件
- 如何生成结构图日志
- 如何生成标量数据日志
- 如何生成媒体数据日志
- 如何生成统计分析数据日志
- 如何生成超参分析日志
- 如何生成降维分析日志
- 如何生成异常检测日志

本文中提到的可视化日志具体使用方式可参考 [test_summary.py](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/python/test/ops/test_summary.py) 文件，

具体可视化效果参考[之江天枢人工智能开源平台](http://tianshu.org.cn/?/course)用户手册可视化部分。

## 初始化

首先定义一个用于存放日志文件的目录 logdir, 我们定义以下作业函数，在其中调用 `flow.summary.create_summary_writer` 接口创建日志文件写入方法。
```python
@flow.global_function(function_config=func_config)
def CreateWriter():
    flow.summary.create_summary_writer(logdir)
```
再调用 `CreateWriter` 这个函数，就可以完成 `summary writer` 对象的创建。

## 生成结构图日志

首先通过调用 `flow.summary.Graph` 接口生成 graph 对象，然后再通过调用 graph 对象的 `write_structure_graph` 方法将 graph 写入到日志文件。

```python
graph = flow.summary.Graph(logdir)
graph.write_structure_graph()
```

## 生成标量数据日志

通过在作业函数（下例中的 `ScalarJob`）调用 `flow.summary.scalar` 接口来创建标量数据写入方法。

```python
@flow.global_function(function_config=func_config)
def ScalarJob(
    value: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.float),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((1000,), dtype=flow.int8),
):
    flow.summary.scalar(value, step, tag)
```

再调用作业函数 `ScalarJob`，就可以将标量数据写入日志文件。

```python
value = np.array([1], dtype=np.float32)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("scalar", dtype=np.int8)
ScalarJob([value], [step], [tag])
```

## 生成媒体数据日志

目前已经支持图片和文本两种媒体数据。

#### 文本数据日志生成

通过在作业函数中调用 `flow.summary.pb` 接口来创建文本数据写入方法。

```python
@flow.global_function(function_config=func_config)
def PbJob(
    value: flow.typing.ListNumpy.Placeholder((1500,), dtype=flow.int8),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
):
    flow.summary.pb(value, step=step)
```

使用的时候，首先我们要定义一个字符串列表，然后通过调用 `flow.summary.text` 接口生成 `protobuf message`，再将其转化成字符串，最后调用作业函数 `PbJob` 将文本数据写入日志文件。

```python
net = ["vgg16", "resnet50", "mask-rcnn", "yolov3"]
pb = flow.summary.text(net)
value = np.fromstring(str(pb), dtype=np.int8)
step = np.array([i], dtype=np.int64)
PbJob([value], [step])
```

#### 图像数据日志生成

通过在作业函数中调用 `flow.summary.images` 接口来创建图片数据写入方法。

```python
@flow.global_function(function_config=func_config)
def ImageJob(
    value: flow.typing.ListNumpy.Placeholder(
        shape=(100, 2000, 2000, 4), dtype=flow.uint8
    ),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((10,), dtype=flow.int8),
):
    flow.summary.image(value, step, tag)
```

再调用作业函数 `ImageJob`，就可以将图片数据写入日志文件，注意我们这里的图片是 RGB 三通道的 jpg 格式。
```python
import cv2
def _read_images_by_cv(image_files):
    images = [
        cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.uint8)
        for image_file in image_files
    ]
    return [cv2.resize(image, (512, 512)) for image in images]
image1_path = "~/oneflow/image1"
image2_path = "~/oneflow/image2"
image_files = [
    image1_path,
    image2_path,
]
images = _read_images_by_cv(image_files)
images = np.array(images, dtype=np.uint8)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("image", dtype=np.int8)
ImageJob([images], [step], [tag])

```

## 生成统计分析数据日志

通过在作业函数中调用 `flow.summary.histogram` 接口来创建统计分析数据写入方法。

```python
@flow.global_function(function_config=func_config)
def HistogramJob(
    value: flow.typing.ListNumpy.Placeholder((200, 200, 200), dtype=flow.float),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((9,), dtype=flow.int8),
):
    flow.summary.histogram(value, step, tag)
```

调用作业函数 `HistogramJob` 时，传入 value, step 和 tag， 即可将统计分析数据写入日志文件。

```python
value = np.random.rand(100, 100, 100).astype(np.float32)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("histogram", dtype=np.int8)
HistogramJob([value], [step], [tag])
```

## 生成超参分析日志

首先，创建一个包含超参数据的字典（以下代码中的 `hparams`），它的 key 需要使用接口 `flow.summary.HParam` 及 `flow.summary.Metric` 创建，value 为超参值。，然后通过 `flow.summary.hparams` 函数生成protobuf message，转化成字符串之后调用作业函数 `PbJob` 将超参分析数据写入日志文件。

```python
@flow.global_function(function_config=func_config)
def PbJob(
    value: flow.typing.ListNumpy.Placeholder((1500,), dtype=flow.int8),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
):
    flow.summary.pb(value, step=step)

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

首先通过调用 `flow.summary.Projector()` 接口生成 `projector` 对象，然后通过 `projecotr`。`create_embedding_projector` 函数创建 `embedding_projector` 对象，最后通过调用 `projecotr.embedding_projector` 函数将降维分析日志写入日志文件。注意 `value` 和 `label` 是需要降维的数据点信息，`x` 代表数据库，`sample_name` 和 `sample_type` 是`x` 里面数据的属性。

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

首先通过调用 `flow.summary.Projector` 接口生成 `projector` 对象，然后通过 `projecotr.create_exception_projector` 函数创建 `exception_projector` 对象，最后通过调用 `projecotr.exception_projector` 函数将异常检测日志写入日志文件。

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
