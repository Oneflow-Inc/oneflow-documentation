#  SAVE AND LOAD THE MODEL

There are two common uses for loading and saving models:

- Save the model that has been trained to continue training next time.
- Save the trained model for direct prediction in the future.

We will introduce how to use [save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save) and [load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) to save and load models as follows.

Also, we will show how to load a pre-trained model to run model predictions.

## Saving and Loading Model Parameters
`Module` provided by OneFlow and defined by users provides the `state_dict` method to obtain all the model parameters, which is stored in a "parameter name-parameter value" dictionary.【不确定】

```python
import oneflow as flow
m = flow.nn.Linear(2,3)
print(m.state_dict())
```

The code above prints out the parameters in m which is in the pre-constructed Linear Module.

```text
OrderedDict([('weight',
              tensor([[-0.4297, -0.3571],
                      [ 0.6797, -0.5295],
                      [ 0.4918, -0.3039]], dtype=oneflow.float32, requires_grad=True)),
             ('bias',
              tensor([ 0.0977,  0.1219, -0.5372], dtype=oneflow.float32, requires_grad=True))])
```

We can load parameters by calling `load_state_dict` method of `Module`, as the following code:

```python
myparams = {"weight":flow.ones(3,2), "bias":flow.zeros(3)}
m.load_state_dict(myparams)
print(m.state_dict())
```

The tensor in the dictionary created by us has been loaded into m Module:

```text
OrderedDict([('weight',
              tensor([[1., 1.],
                      [1., 1.],
                      [1., 1.]], dtype=oneflow.float32, requires_grad=True)),
             ('bias',
              tensor([0., 0., 0.], dtype=oneflow.float32, requires_grad=True))])
```


## Saving Models
We can use [oneflow.save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save) to save models.

```python
flow.save(m.state_dict(), "./model")
```

The first parameter is the Module parameters, and the second is the saved path. The above code saves the parameters of the `m` Module object to the path `./model`.



## Loading Models

Using [oneflow.load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) to load parameters from specified disk path to the memory, and get the dictionary with the parameters.

```python
params = flow.load("./model")
```

Then use `load_state_dict` to load the dictionary into the model.

```python
m2 = flow.nn.Linear(2,3)
m2.load_state_dict(params)
print(m2.state_dict())
```

We have created a new Linear Module object `m2`, and loaded the parameters saved from the above to `m2`. Then we get the output as below:

```text
OrderedDict([('weight', tensor([[1., 1.],
        [1., 1.],
        [1., 1.]], dtype=oneflow.float32, requires_grad=True)), ('bias', tensor([0., 0., 0.], dtype=oneflow.float32, requires_grad=True))])
```


### Using a Pre-trained Model to Make Predictions

OneFlow can directly load PyTorch's pre-trained model for prediction as long as the structure and parameter names of the model are aligned with the PyTorch model.

Examples can be found in [here](https://github.com/Oneflow-Inc/models/tree/main/shufflenetv2#convert-pretrained-model-from-pytorch-to-oneflow).

Run commands below for trying how to use the pre-trained model to make predictions:

```bash
git clone https://github.com/Oneflow-Inc/models.git
cd models/shufflenetv2
bash infer.sh
```
