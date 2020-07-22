In this article, we will learn:

* Using OneFlow's port to configure software and hardware environment.

* Using OneFlow's port to define training model.

* Achieved OneFlow's training job function.

* Save/load the result of training model.

* Achieved OneFlow's evaluation of job function.

This article demonstrated the core step of OneFlow by using the LeNet model to training MNIST dataset. The full example code is attached in the end.

Before learning, you can check the function of each script by running the following command.

First of all, clone the documentation repository and switch to the corresponding path:
```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd oneflow-documentation/docs/code/quick_start/
```

* Training model
```shell
python lenet_train.py
```
The command above will perform traning of MNIST dataset and saving the model.

Training model is the precondition of `lenet_eval.py` and `lenet_test.py`. Or we can directly download and use our model which is already been trained and skip the training progress:
```shell
#Repository location: docs/code/quick_start/ 
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
```

* Model evaluation
```shell
python lenet_eval.py
```
The command above using the MNIST's testing set to evaluate the training model and print the accuracy.

* Image recognition

```shell
python lenet_test.py ./9.png
```
The above command will using the training model we just saved to predicting the content of "9.png". Or we can download our [ prepared image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)to verify their training model prediction result.

## MNIST dataset introdaction

MNIST is a handwritten number database which including training set and testing set.Training set include 60000 pictures and the corresponding label.Yann LeCun and etc... has normalise the image size and pack them to binary file for download.http://yann.lecun.com/exdb/mnist/

## Configuration of hardware and software training environment

Using oneflow.function_config() can construct a configuration object. By using that object, we can configure many hardware and software parameters relevant to training. The parameters directly relating to the training is been packed and store in `function_config`'s train members. The rest of configuration directly set as the members of `function_config`. Following is our basic configuration of training:

```python
def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config
```

In the code above:

* We put the default type of training as float

* We set learning rate as 0.1

* We set update strategy as "naive_conf" during the training

config object and it's usage scenarios will be introduce in **Implement training function** later.

## Define training model

In oneflow.nn and oneflow.layers, provide the operator to used to construct the model.

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')

```

In the code above, we build up a LeNet network model.

## Implement training function

OneFlow provide a decorator called `oneflow.global_function`. By using it, we can covert a Python function to job function.

### Function decorator

`oneflow.global_function` decorator receive a `function_config` object as parameter. It can can covert a normal Python function to job function of OneFlow and use the configuration we just done for `function_config`.

```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
    #任务函数实现 ...
```

### Specify the optimization feature
We can using `oneflow.losses.add_loss`'s port to specify the parameters which need to optimization.We can using `oneflow.losses.add_loss`'s port to specify the parameters which need to optimization.In this way, OneFlow will trade optimise the parameter as target when each iteration training mission.

```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss
```

So Far, we using `flow.nn.sparse_softmax_cross_entropy_with_logits` to calculate the loss and trade optimize loss as target parameter.

** Attention **: The training function use add_loss to specified optimize parameters. It is irrelevant with return value. It is irrelevant with return value. The return value in the demonstration above is loss but not necessary. In fact, the reture value of training function is for Interactive with external environment during the training.We will introduce that in more details in **Call the job function and interaction**below.

## Call the jon function and interaction

We can start training when called the job function.Thus, the return value is OneFlow encapsulated object **rather than **previously defined job function's return values. This involved a issue which how to interact **previous defined** job function and **previous called** job function. This involved a issue which how to interact **previous defined** job function and **previous called** job function. The solution is sample, when **called **the return object. It included `get`and `async_ge` method. They corresponding to synchronous and asynchronous.Through them, we can obtained return value of function when **defined them**.Through them, we can obtained return value of function when **defined them**.

### Synchronously method obtained the return value when training task

By using get method, we can synchronous obtain the return value.

```python
  for epoch in range(50):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get().mean()
      if i % 20 == 0: print(loss)
```

The code above, using `get` method to obtain the loss vector and calculated the average then print it.

### Asynchronous method obtained the return value when training task

`async_get` is use for asynchronous get the train job function's return value when **defined it**. It need us to prepare a callback function. When OneFlow finish iterate thire train job function, it will call our callback function and sned the return value of train job function as parameter to our callback function. Sample code:

```python
cb_handle_result(result):
    #... job_func(images, labels).async_get(cb_handle_result)
```

More details example will be demonstrated in **Model evaluation**.

## Initialization, saving and loading models

### Initialization and saving model

The object structured by `oneflow.train.CheckPoint` can use for initialization, saving and loading models. During the training process, we can use `init` to initialize model and use `save` to save model.For example:For example:

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.init()
  #load data and training ...  
  check_point.save('./lenet_models_1') 
```

When save successfully, we will get a ** directory** called "lenet_models_1". This directory included directories and files corresponding with the model parameters.

### Loading models

During the evaluation or prediction, we can use `oneflow.train.CheckPoint.load` to load the existing model parameters.For example:For example:

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.load("./lenet_models_1")
  #evaluation process  ...
```

Automatically load the model we saved previously.

## Evaluation of models
Evaluation job function **basically is same as** train job function. The difference is in evaluation process, the model we use is already saved. Thus, do not require initialize and update model during Iteration.

### Configure the hardware and software environment of evaluation

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config
```

Above code is the configuration of function_config during the evaluation. Compare with the training process, cut of the option in learning rate and the settings of update model parameters.

### Coding of evaluation job function

```python
@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

Above is the coding of evolution job function and return object is a dictionary.Above is the coding of evolution job function and return object is a dictionary.We will call train job function and demonstrated how to use asynchronous method to obtain return value.

### Iteration evaluation

Prepare callbcak function:
```python
g_total = 0
g_correct = 0

def acc(eval_result):
  global g_total
  global g_correct

  labels = eval_result["labels"]
  logits = eval_result["logits"]

  predictions = np.argmax(logits.ndarray(), 1)
  right_count = np.sum(predictions == labels)
  g_total += labels.shape[0]
  g_correct += right_count
```

The callbcak function above is `acc`. It will be call by OneFlow frame. All the parameters obtained(`eval_result`) is the return value of train job function. We record the total number of sample and total correct results number of sample in this function.

Called evacuation job function:

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.load("./lenet_models_1")
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE, BATCH_SIZE)
  for epoch in range(1):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      eval_job(images, labels).async_get(acc)

  print("accuracy: {0:.1f}%".format(g_correct*100 / g_total))
```

So far, Cycle call the evaluation function and output the accuracy of result of testing set.

## Image prediction

Modify the above evaluation code, change the evaluate date to raw images rather than the existing dataset. Then we can use model to predict the content in the image.

```python
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im -128.0)/ 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im

if __name__ == '__main__':
    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")

    image = load_image(sys.argv[1])
    logits = eval_job(image, np.zeros((1,)).astype(np.int32)).get()

    prediction = np.argmax(logits.ndarray(), 1)
    print("predict:{}".format(prediction))
```

## Complete code

### Training model

Name: [lenet_train.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_train.py)

```python
#lenet_train.py
import numpy as np
import oneflow as flow
from mnist_util import load_data
from PIL import Image
BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((1, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((1,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
    return logits

if __name__ == '__main__':
    flow.config.gpu_device_num(1)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)

    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels).get().mean()
            if i % 20 == 0: print(loss)
            if loss < 0.01:
                break
    check_point.save('./lenet_models_1')  # need remove the existed folder
    print("model saved")
```

### Evaluate model

Name: [lenet_eval.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_eval.py)

Saved model: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

```python
#lenet_eval.py
import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return {"labels": labels, "logits": logits}


g_total = 0
g_correct = 0


def acc(eval_result):
    global g_total
    global g_correct

    labels = eval_result["labels"]
    logits = eval_result["logits"]

    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


if __name__ == '__main__':

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE, BATCH_SIZE)

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            eval_job(images, labels).async_get(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

### Number prediction

Name: [lenet_test.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_test.py)

Saved model: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

MNIST image dataset [mnist_raw_images.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)

```python
import numpy as np
import oneflow as flow
from PIL import Image

BATCH_SIZE = 1


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
    return logits


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


if __name__ == '__main__':

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")

    image = load_image("./9.png")
    logits = eval_job(image, np.zeros((1,)).astype(np.int32)).get()

    prediction = np.argmax(logits.ndarray(), 1)
    print("predict:{}".format(prediction[0]))
```
