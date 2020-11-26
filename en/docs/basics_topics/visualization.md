# Visualization
In OneFlow, we support to save a summary of data used during the training procedure into a local log file. Therefore, we can visualize the real time training data in a visualization frontend by reading the log file in the visualization backend.

Currently, our visualization support the following types of data

| Visualization Type | Description                     |
| ---------- | ------------------------ |
| Model Structure   | Logical Graph, Physical Graph (will come up in later version) |
| Scalar Data   | Scalar type of data                |
| Media Data   | Text, images               |
| Statistical Analysis   | Histogram、scatter diagram   |
| Hyper-parameters   | Visualize and analysis  of hyper-parameters                   |
| Projection Embedding   | Data projection from high dimensional data to lower dimensional data             |
| Exception Detection   | Dectection of exceptional data             |

In this page, we shall introduce

* how to initialize a summary writer;

* how to generate a model graph summary;

* how to generate a scalar summary;

* how to generate a summary for media data;

* how to generate summary for statistical analysis;

* how to generate summary for hyper-parameters; and

* how to generate summary for projection embedding;

* how to generate summary for exception detection.

The usage of summary in this page refers to test_summary.py,

and the effect of visualization backend and frontend refers to the documentation of [Zhijiang Tianshu Open Source AI Platform](http://tianshu.org.cn/?/course). 

## Initialization

First define a file directory for saving the log logdir, then we define a function to create the log file by using the api flow.summary.create_summary_writer.
```python
@flow.global_function(function_config=func_config)
def CreateWriter():
    flow.summary.create_summary_writer(logdir)
```
Thus we can create an object of summary writer by CreateWriter().



## Structural Graph Summary

First we call the api flow.summary.Graph to generate a graph object, then we use the api graph.write_structure_graph to write the graph into the log file.

```python
graph = flow.summary.Graph(logdir)
graph.write_structure_graph()
```



## Scalar Summary

First we define a scalar data writing function by the api flow.summary.scalar.

```python
@flow.global_function(function_config=func_config)
def ScalarJob(
    value: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.float),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((1000,), dtype=flow.int8),
):
    flow.summary.scalar(value, step, tag)
```

Then we call ScalarJob() to write scalar data into the log file.

```python
value = np.array([1], dtype=np.float32)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("scalar", dtype=np.int8)
ScalarJob([value], [step], [tag])
```



## Media Data Summary

Here we support two types of media data, namely, text and images.

####  Text Summary

First we define a text data writing function by the api flow.summary.pb.

```python
@flow.global_function(function_config=func_config)
def PbJob(
    value: flow.typing.ListNumpy.Placeholder((1500,), dtype=flow.int8),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
):
    flow.summary.pb(value, step=step)
```

We write the text into a list of strings. Then we create a protobuf message by using the api flow.summary.text. Finally, we translate the protobuf message into a string and use PbJob() to write the text data in to the file.

```python
net = ["vgg16", "resnet50", "mask-rcnn", "yolov3"]
pb = flow.summary.text(net)
value = np.fromstring(str(pb), dtype=np.int8)
step = np.array([i], dtype=np.int64)
PbJob([value], [step])
```

#### Image Summary

We define image summary writing function by the api flow.summary.images.

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

Then we call the function ImageJob()  to write the image data into the file. Please note that we need to use the jpg image in RGB format.
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



## Statistical Summary

We create the statistical summary writing method by the api flow.summary.histogram

```python
@flow.global_function(function_config=func_config)
def HistogramJob(
    value: flow.typing.ListNumpy.Placeholder((200, 200, 200), dtype=flow.float),
    step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    tag: flow.typing.ListNumpy.Placeholder((9,), dtype=flow.int8),
):
    flow.summary.histogram(value, step, tag)
```

Then by giving value, step and tag, we can call the function HistogramJob() to write statistical data into the log file.

```python
value = np.random.rand(100, 100, 100).astype(np.float32)
step = np.array([1], dtype=np.int64)
tag = np.fromstring("histogram", dtype=np.int8)
HistogramJob([value], [step], [tag])
```



## Hyper-parameters

First we create map harams that contains the hyper-parameters, then we create the protobuf message using the api flow.summary.hparams. After that, we translate the message into a string and call the function PbJob() to write the hyper-parameters into the log file.

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



## Embedding Projection

First, we create a projector object using flow.summary.Projector(). Then we create an embedding_projector object by the api projecotr.create_embedding_projector(). Finally we call the function projecotr.embedding_projector() to write the embedding projector analysis into the log file. Note that in the embedding projector, value and label refers to the data, x refers to the data set, sample_name and sample_type refers to the type of data in x.

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



## Exception Detection

First, we create a projector object using flow.summary.Projector(). Then we create an exception_projector object by the api projecotr.create_exception_projector(). Finally we call the function projecotr.exception_projector() to write the exception detection analysis into the log file.

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

