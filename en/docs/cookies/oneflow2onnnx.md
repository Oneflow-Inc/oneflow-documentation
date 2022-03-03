# Onelow with ONNX

This document introduces the usage of OneFlow interacting with ONNX, including how to export OneFlow models to ONNX, and how to use ONNX models for inference.


## Introduction to ONNX

[ONNX](https://onnx.ai/index.html), known as Open Neural Network Exchange, is an open file format standard designed for machine learning algorithms to store trained algorithmic models. Many major deep learning frameworks (e.g., OneFlow, PyTorch, TensorFlow, MXNet) support exporting models to ONNX models, which allows different deep learning frameworks to store and interact with model data in a uniform format. In addition, ONNX has a corresponding Runtime - [ONNX Runtime](https://onnxruntime.ai/) - that facilitates model deployment and reasoning on multiple platforms (Linux, Windows, Mac OS, Android, iOS, etc.) and multiple hardware (CPU, GPU, etc.). 

### Related Packages

There are several ONNX-related libraries, and the features of several common libraries are described below. The onnxruntime-gpu involved in this tutorial can be installed via `pip install onnxruntime-gpu`.

1. [onnx](https://github.com/onnx/onnx): ONNX model format standard

2. [onnxruntime & onnxruntime-gpu](https://github.com/microsoft/onnxruntime): ONNX runtime that is used to load the ONNX model for inference. onnxruntime and onnxruntime-gpu support CPU inference and GPU inference respectively.

3. [onnx-simplifier](https://github.com/daquexian/onnx-simplifier): for simplifying ONNX models, e.g. eliminating operators with constant results
   
4. [onnxoptimizer](https://github.com/onnx/optimizer): it is used to optimize ONNX model by graph transformations


## Export OneFlow Models to ONNX Models

[oneflow-onnx](https://github.com/Oneflow-Inc/oneflow_convert) is a model conversion tool provided by OneFlow team to support exporting OneFlow static graph models to ONNX models. At present oneflow-onnx supports more than 80 kinds of OneFlow OPs exported as ONNX OPs. For detalis, refer to [list of OP supported by OneFlow2ONNX](https://github.com/Oneflow-Inc/oneflow_convert/blob/main/docs/oneflow2onnx/op_list.md)。

### Install oneflow-onnx

oneflow-onnx is independent of OneFlow and needs to be installed separately via pip:

```bash
pip install oneflow-onnx
```

### How to Use oneflow-onnx

To export OneFlow static graph model as ONNX model, just call `export_ onnx_ Model` function.

```python
from oneflow_onnx.oneflow2onnx.util import export_onnx_model

export_onnx_model(graph,
                  external_data=False, 
                  opset=None, 
                  flow_weight_dir=None, 
                  onnx_model_path="/tmp", 
                  dynamic_batch_size=False)
```

The meaning of each parameter is as follows:

1. graph: the graph need to be converted ([Graph](../basics/08_nn_graph.md) object)

2. external_data: whether to save the weights as external data of the ONNX model. When it is `True`, it is usually to avoid the 2GB file size limit of protobuf.

3. opset: specify the version of the conversion model (int, default is 10)

4. flow_weight_dir: path to save OneFlow model weights

5. onnx_model_path: save path for exported ONNX models

6. dynamic_batch_size: whether the exported ONNX model supports dynamic batch, default is False


In addition, oneflow-onnx provides a function called `convert_to_onnx_and_check` to convert and meanwhile check the converted ONNX model. The check process will pass the same input to the original OneFlow model and the converted ONNX model respectively, and then compare the difference between each value in the two outputs to see if they are same within a relative range.

```python
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

convert_to_onnx_and_check(...)
```

The parameters of the `convert_to_onnx_and_check` are almost the same as those of `export_onnx_model`, besides  you can pass `print_outlier` parameter additionally. When `print_outlier=True`, it will output any abnormal values found during the check process that exceed the reasonable error range. 

### Considerations when Exporting Models

- Before exporting the model, it need be set to eval mode because operations such as Dropout and Batch Normalization have different behaviors under the training and evaluation mode.
- When building a static graph model, you need to specify an input. The value of the input can be random, but make sure the data type and shape is correct.
- The ONNX model accepts a fixed shape of input, and a varied size of the batch dimension, so by setting the `dynamic_batch_size` parameter to be `True` can make the exported ONNX model support dynamic batch size.
- Oneflow-onnx must use a static graph model (Graph mode) as an parameter to export function. For dynamic graph models (Eager mode), the dynamic graph model needs to be constructed as a static graph model. Refer to the example below.


## Examples of Usage

In this section, the process of exporting a OneFlow model to an ONNX model and performing inference is introduced, using the common ResNet-34 model as an example.

The following code uses [FlowVision](https://github.com/Oneflow-Inc/vision), a library built on OneFlow for computer vision tasks, which contains many models, data enhancement methods, data transformation operations, datasets, and so on. Here we directly use the ResNet-34 model provided by the FlowVision library and use its weight trained on the ImageNet dataset. 

### Export as ONNX Model

Import related dependencies:

```python
import oneflow as flow
from oneflow import nn
from flowvision.models import resnet34
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
```

To build a static graph model using a dynamic graph model. For details, refer to: [Static Graph Interface: nn.Graph](../basics/08_nn_graph.md)

```python
class ResNet34Graph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(self, x):
        return self.model(x)
```

Export OneFlow static graph models to ONNX models:

```python
# Model parameter storage directory
MODEL_PARAMS = 'checkpoints/resnet34'

params = flow.load(MODEL_PARAMS)
model = resnet34()
model.load_state_dict(params)

# Set the model to eval mode
model.eval()

resnet34_graph = ResNet34Graph(model)
# Build the static graph model
resnet34_graph._compile(flow.randn(1, 3, 224, 224))

# Export as ONNX model and check
convert_to_onnx_and_check(resnet34_graph, 
                          flow_weight_dir=MODEL_PARAMS, 
                          onnx_model_path="./", 
                          print_outlier=True,
                          dynamic_batch_size=True)
```

After running,  a file named `model.onnx` is  in the current directory, which is the exported ONNX model.

### Inference with ONNX models

Before performing inference, ensure that the ONNX Runtime is installed, that is onnxruntime or onnxruntime-gpu. In the experimental environment of this tutorial, onnxruntime-gpu is installed to invoke the GPU for computation, but if there is no GPU on the machine, you can specify the CPU for calculation. See below for details.

We use the following image as input to the model:
<div align="center">
    <img alt="Demo Image" src="./imgs/cat.jpg" width="300px">
</div>


Import related dependencies:

```python
import numpy as np
import cv2
from onnxruntime import InferenceSession
```

Define a function to pre-process the image to a format and size accepted by the ONNX model:

```python
def preprocess_image(img, input_hw = (224, 224)):
    h, w, _ = img.shape
    
    # Use the longer side of the image to determine the scaling factor
    is_wider = True if h <= w else False
    scale = input_hw[1] / w if is_wider else input_hw[0] / h

    # Scale the image equally
    processed_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # Normalization
    processed_img = np.array(processed_img, dtype=np.float32) / 255

    # Fill images to ONNX model and preset sizes
    temp_img = np.zeros((input_hw[0], input_hw[1], 3), dtype=np.float32)
    temp_img[:processed_img.shape[0], :processed_img.shape[1], :] = processed_img
    processed_img = temp_img
    
    # Adjust the order of axes and add batch axes at the front 
    processed_img = np.expand_dims(processed_img.transpose(2, 0, 1), axis=0)

    return processed_img
```

The next step is to use the ONNX model for inference, which consists of creating an InferenceSession object and calling `run` to perform inference.

In onnxruntime(-gpu) 1.9 and above, the `providers` parameter needs to be explicitly specified when creating an InferenceSession object to select the hardware to use. For onnxruntime-gpu, the values that can be specified include `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`. If there is no GPU on the running machine, you can specify the `providers` parameter as `['CPUExecutionProvider']` to use the CPU for computation.

The type of input data of an ONNX model is a dict. Its keys are `input names` when exporting the ONNX model, and the values are the actual input data of NumPy array type. You can get "input names" through the `get_inputs` method of the InferenceSession object, which returns a list of objects of `onnxruntime.NodeArg` type. For NodeArg object, you can use its `name` property to get a name of str type. In this tutorial, the input is only the image data, so you can get the "input names" corresponding to the input by calling `.get_inputs()[0].name` on the InferenceSession object. The value is `_ResNet34Graph_0-input_0/out`, which is used as the key to construct the dict of the ONNX model input. Of course, it can also be obtained dynamically at runtime without specifying it in advance.

```python
# Read the category name of the ImageNet dataset from the file
with open('ImageNet-Class-Names.txt') as f:
    CLASS_NAMES = f.readlines()

# Read the image file and preprocess it with the `preprocess_image` function
img = cv2.imread('cat.jpg', cv2.IMREAD_COLOR)
img = preprocess_image(img)

# Create an InferenceSession object
ort_sess = InferenceSession('model.onnx', providers=['TensorrtExecutionProvider',
                                                     'CUDAExecutionProvider',
                                                     'CPUExecutionProvider'])
# Call the `run` method of the InferenceSession object to perform inference
results = ort_sess.run(None, {"_ResNet34Graph_0-input_0/out": img})

# Output inference results
print(CLASS_NAMES[np.argmax(results[0])])
```

The output of the `run` method of the InferenceSession object is a list of NumPy arrays, and each NumPy array corresponds to a set of outputs. Since there is only one set of inputs, the element with index 0 is the output, and the shape of it is `(1, 1000)`, which corresponds to the probability of 1000 categories (if n images are input as a batch, the shape of them will be `(n, 1000)`). After obtaining the index corresponding to the category with the highest probability via `np.argmax`, the index is mapped to the category name.

Run the code and get the result:

```text
(base) root@training-notebook-654c6f-654c6f-jupyter-master-0:/workspace# python infer.py 
285: 'Egyptian cat',
```

The above inference is done in a Python environment using GPU or CPU. In practice, you can use the exported ONNX model with a different ONNX Runtime depending on the deployment environment.
