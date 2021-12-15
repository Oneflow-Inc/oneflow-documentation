# 模型的加载与保存

对于模型的加载与保存，常用的场景有：

- 将已经训练一段时间的模型保存，方便下次继续训练
- 将训练好的模型保存，方便后续直接用于预测

在本文中，我们将介绍，如何使用 [save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save) 和 [load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) API 保存模型、加载模型。

同时也会展示，如何加载预训练模型，完成预测任务。

## 模型参数的获取与加载

OneFlow 预先提供的各种 `Module` 或者用户自定义的 `Module`，都提供了 `state_dict` 方法获取模型所有的参数，它是以 “参数名-参数值” 形式存放的字典。

```python
import oneflow as flow
m = flow.nn.Linear(2,3)
print(m.state_dict())
```

以上代码，将显式构造好的 Linear Module 对象 m 中的参数打印出来：

```text
OrderedDict([('weight',
              tensor([[-0.4297, -0.3571],
                      [ 0.6797, -0.5295],
                      [ 0.4918, -0.3039]], dtype=oneflow.float32, requires_grad=True)),
             ('bias',
              tensor([ 0.0977,  0.1219, -0.5372], dtype=oneflow.float32, requires_grad=True))])
```

通过调用 `Module` 的 `load_state_dict` 方法，可以加载参数，如以下代码：

```python
myparams = {"weight":flow.ones(3,2), "bias":flow.zeros(3)}
m.load_state_dict(myparams)
print(m.state_dict())
```

可以看到，我们自己构造的字典中的张量，已经被加载到 m Module 中：

```text
OrderedDict([('weight',
              tensor([[1., 1.],
                      [1., 1.],
                      [1., 1.]], dtype=oneflow.float32, requires_grad=True)),
             ('bias',
              tensor([0., 0., 0.], dtype=oneflow.float32, requires_grad=True))])
```

## 模型保存

我们可以使用 [oneflow.save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save)方法保存模型。

```python
flow.save(m.state_dict(), "./model")
```

它的第一个参数的 Module 的参数，第二个是保存路径。以上代码，将 `m` Module 对象的参数，保存到了 `./model` 目录下。

## 模型加载

使用 [oneflow.load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) 可以将参数从指定的磁盘路径加载参数到内存，得到存有参数的字典。

```python
params = flow.load("./model")
```

然后，再借助上文介绍的 `load_state_dict` 方法，就可以将字典加载到模型中：

```python
m2 = flow.nn.Linear(2,3)
m2.load_state_dict(params)
print(m2.state_dict())
```

以上代码，新构建了一个 Linear Module 对象 `m2`，并且将从上文保存得到的的参数加载到 `m2` 上。得到输出：

```text
OrderedDict([('weight', tensor([[1., 1.],
        [1., 1.],
        [1., 1.]], dtype=oneflow.float32, requires_grad=True)), ('bias', tensor([0., 0., 0.], dtype=oneflow.float32, requires_grad=True))])
```

### 使用预训练模型进行预测

OneFlow 是可以直接加载 PyTorch 的预训练模型，用于预测的。
只要模型的作者能够确保搭建的模型的结构、参数名与 PyTorch 模型对齐。

相关的例子可以在 [OneFlow Models 仓库的这个 README](https://github.com/Oneflow-Inc/models/blob/main/README_zh-CN.md) 查看。

以下命令行，可以体验如何使用预训练好的模型，进行预测：

```bash
git clone https://github.com/Oneflow-Inc/models.git
cd models/shufflenetv2
bash infer.sh
```
