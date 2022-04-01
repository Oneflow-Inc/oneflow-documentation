
# Converting Pre-trained Model from PyTorch to OneFlow


Since interfaces of OneFlow model and PyTorch model is compatible, you can convert a pre-trained model from PyTorch to OneFlow when you need to use a PyTorch pre-trained model.

## Example of model conversion

In the following code, we will define and save a PyTorch model and then convert it to a OneFlow model. 

```python
import torch
import torch.nn as nn
save_file = 'model.pth'
model_torch = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)
torch.save(model, save_file)
```

After running the above code, we get a  `model.pth` file of PyTorch model. Then, the following two steps enable us to covert a PyTorch model to a OneFlow model：


- defining a OneFlow model with **the same structure**
- loading the `model.pth` file and initializing model parameters into OneFlow model

Code is shown below:

```python
import oneflow as flow
import oneflow.nn as nn
import torch
model_flow = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)
parameters = torch.load(save_file).state_dict()
for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val
model_flow.load_state_dict(parameters)
```


`.state_dict()` enables to obtain model parameters defined by `key-value` . Then, we use `.detach().cpu().numpy()` to convert parameter value that gradients are blocked into Numpy. Lastly,  `.load_state_dict(parameters)` allows to pass model parameters to OneFlow model. 


With the simple example described above, we can find that the approach to convert data that is saved by PyTorch model into OneFlow is to **use Numpy as a bridge**. Therefore, provided the models defined by PyTorch and by OneFlow are compatible, even complicated models can still be smoothly converted.


## More information about flowvision


Same as torchvision, [flowvision](https://github.com/Oneflow-Inc/vision) also provides many pre-trained models, and the models in flowvision are compatible with those in torchvision. Taking AlexNet for example, flowvision enables us to convert **complicate PyTorch pre-trained models** into OneFlow by running the following code: 

```python
import torchvision.models as models_torch
import flowvision.models as models_flow
alexnet_torch = models_torch.alexnet(pretrained=True)
alexnet_flow = models_flow.alexnet()
parameters = alexnet_torch.state_dict()
for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val
alexnet_flow.load_state_dict(parameters)
```


You can also use pre-trained models provided in flowvision by importing the following code:

```python
alexnet_flow = models_flow.alexnet(pretrained=True)
```



For more information about flowvision, please visit [flowvision documentation](https://flowvision.readthedocs.io/en/latest/index.html). 
