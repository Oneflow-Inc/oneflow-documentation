This article covers topics below:

* Configuring the hardware and software environment using the OneFlow interface

* Define models using OneFlow's interface

* Model training with `train` type

* How to save/load model

* Use the `predict` type for model evaluation 

* Using `predict` type for image recognition

This article demonstrates the key steps of how to train a LeNet model with MNIST dataset using OneFlow. The full example code is attached at the end of article.

You can see the effects of each script by running the following commands (GPU device is required).

First of all, clone the documentation repository and switch to the corresponding path:
```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd oneflow-documentation/en/docs/code/quick_start/
```

* Training model
```shell
python lenet_train.py
```
The commands above will train a model with MNIST dataset and save it.

Output：

```shell
File mnist.npz already exist, path: ./mnist.npz
5.9947124
1.0865117
0.5317516
0.20937675
0.26428983
0.21764673
0.23443426
...
```

> A trained model is the prerequisite of `lenet_eval.py` and `lenet_test.py`. We can directly download a trained model to skip the training progress:

```shell
#change directory to: en/docs/code/quick_start/ 
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
```

* Evaluation
```
python lenet_eval.py
```
The command above uses the MNIST's testing set to evaluate the trained model and print out the accuracy.

Output：

```text
File mnist.npz already exist, path: ./mnist.npz
accuracy: 99.4%
```

* Image recognition

```shell
python lenet_test.py ./9.png
# Output：prediction: 9
```
The above command will use the trained model to predict the content of file "9.png". We can also download and verify more from [prepared images](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip).

## Introduction of MNIST Dataset 

MNIST is a handwritten digits database including training set and testing set. Training set includes 60000 pictures and their corresponding label. Yann LeCun and others have normalized all the images and packed them into a single binary file for downloading. http://yann.lecun.com/exdb/mnist/


## Define Training Model

Modules [oneflow.nn](https://oneflow.readthedocs.io/en/master/nn.html) and [oneflow.layers](https://oneflow.readthedocs.io/en/master/layers.html) provide the operators to construct the model.

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")
```


As the code showing above, we build up a LeNet network model.

## Implementation of Job Function for Training

OneFlow provides a decorator named [oneflow.global_function](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.global_function)  by which we can covert a Python function to a OneFlow **Job Function** .

### `global_function` Decorator

 [oneflow.function_config](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.python.framework.function_util.FunctionConfig#oneflow.FunctionConfig) decorator takes a `type` parameter to specify the type of job function. The `type="tranining"` means that the job function is for traning and `type="predict"` is for predicting. 

There is also a `function_config` parameter taken by `oneflow.global_function` decorator. The `function_config` contains configuration about training.

```python
@flow.global_function(type="train")
def train_job(images:tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    # Implementation of netwrok ...
```

The `tp.Numpy.Placeholder` is a placeholder. The annotation `tp.Numpy` on return type means that the job function will return a `numpy` object.

### Setup Optimizer
We can use [oneflow.optimizer](https://oneflow.readthedocs.io/en/master/optimizer.html) to specify the parameters needed by optimization. By this way, in the process of each iteration during training, OneFlow will take the specified object as optimization goal.

```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```

So Far, we use `flow.nn.sparse_softmax_cross_entropy_with_logits` to calculate the loss and specify it as optimization goal.



 - **lr_scheduler** sets the learning rate schedule, and `[0.1]` means learning rate is 0.1.
 -  **flow.optimizer.SGD** means SGD is specified as the optimizer. The `loss` is the goal of minimization to the optimizer and the return type (not requried).
 
## Calling the Job Function and Get Results

We can start training by invoking the job function.

The return value we get when we call the job function is defined by the annotation of return value type in job function. 

We can get one or multiple results after each call of job function.

### Example on Single Return Value
The job function in [lenet_train.py](../code/quick_start/lenet_train.py):
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```
The return value in job function is a `tp.Numpy`. When calling job function, we will get a `numpy` object:
```python
for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

We call the `train_job` and print the `loss` every 20 iterations.

### Example on Multiple Return Values
In script [lenet_eval.py](../code/quick_start/lenet_eval.py), we define the job function below:
```python
@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)
```

The return value type of this job function is `Tuple[tp.Numpy, tp.Numpy]`. When we call the job function, we will get a `tuple` container. There are two `numpy` objects in it:
```python
for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            labels, logits = eval_job(images, labels)
            acc(labels, logits)
```
We call the job function and get `labels` and `logits` then use them to evaluate the model.


### Synchronous and Asynchronous Call
All code in this article only call synchronously to get results from job function. In fact, OneFlow can call job function asynchronously. For more details, please refer to [Obtain results from job function](../basics_topics/async_get.md).


## Model Initialization, Saving and Loading

### Model Initialization and Saving

The example of model saved by the `flow.checkpoint.save`:


```python
if __name__ == '__main__':
  #data loading and training ...  
  flow.checkpoint.save("./lenet_models_1")
```

When the model is saved, we will get a **folder** called "lenet_models_1". This folder contains directories and files corresponding with the model parameters.

### Model Loading

During the prediction process, we can load the parameter from the file to memory by `flow.checkpoint.get` and then update the parameter to the model by `flow.load_variables`. For example:

```python
if __name__ == '__main__':
  flow.load_variables(flow.checkpoint.get("./lenet_models_1"))
  #evaluation process  ...
```

## Evaluation of Model
The job function for evaluation is **basically same** as job function for training. The small difference is that the model we use is already saved in evaluation process. Thus, initialization and update of model during iteration are not needed.

### Job Function for Evaluation
```python
@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)
```

Code above is the implementation of job function for evaluation and its return type is declared as `Tuple[tp.Numpy, tp.Numpy]`. Tuple have two `numpy`  in it. We will call the job function and calculate the accuracy according to the return values.

### Process of Evaluation
The `acc` function is used to count the total number of samples and the number of correct prediction results. We will call the job function to get paramters `labels` and `logits`:
```python
g_total = 0
g_correct = 0


def acc(labels, logits):
    global g_total
    global g_correct

    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count

```
Call the job function for evaluation:

```python
if __name__ == "__main__":
    flow.load_variables(flow.checkpoint.get("./lenet_models_1"))
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            labels, logits = eval_job(images, labels)
            acc(labels, logits)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

So far, we call the job function for evaluation looply and print the accuracy of evaluation result on MNIST testing set.

## Image Prediction
After making a few changes to the code above, it will take the data from the raw images rather than existing dataset. Then we can get a model to predict the content from the images.

```python
def load_image(file):
    im = Image.open(file).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


def main():
    if len(sys.argv) != 2:
        usage()
        return
    flow.load_variables(flow.checkpoint.get("./lenet_models_1"))

    image = load_image(sys.argv[1])
    logits = eval_job(image, np.zeros((1,)).astype(np.int32))

    prediction = np.argmax(logits, 1)
    print("prediction: {}".format(prediction[0]))


if __name__ == "__main__":
    main()
```

## Code

### Model training

Script: [lenet_train.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_train.py)


### Model evaluation

Script: [lenet_eval.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_eval.py)

Saved model: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

### Digits prediction

Script: [lenet_test.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_test.py)

Saved model: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

MNIST image dataset [mnist_raw_images.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)

