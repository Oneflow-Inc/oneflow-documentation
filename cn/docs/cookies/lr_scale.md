# 分层lr_scale设置

通常情况下，在训练神经网络时，只需要设置一个`lr`，例如配置`Adam`优化器

``` python
optimizer = flow.optim.Adam(model.parameters(), lr=1e-3)
```

然而在一些特殊的网络结构中，例如Vision Transformer(ViT)，常常需要在不同的layer设置不同的学习率。

这篇文章以ViT为例讲解oneflow在静态图模式(Graph)和动态图模式(Eager)下如何配置优化器完成不同layer的lr设置。

### Graph模式下的lr_scale设置

先从较为简单的Graph模式开始，首先需要指定每一层的`lr_scale`，存放在`layer_scales`列表中

```python
layer_decay = 0.9
num_layers = len(model.blocks) + 1
layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
```

接下来我们将model的参数进行分组存放，设置每组的`lr_scale`属性

```python
param_groups = {} 
param_group_names = {}
for name, param in model.named_parameters():
    if not param.requires_grad:
    	continue

    layer_idx = get_layer_idx_for_vit(name, num_layers) # 根据参数名找到所在层数
    group_name = "layer_%d" % layer_idx

    if group_name not in param_group_names:
    	this_scale = layer_scales[layer_idx]

        param_group_names[group_name] = {
            "lr_scale": this_scale,
            "params": [],
        }
        param_groups[group_name] = {
            "lr_scale": this_scale, # 设置lr_scale属性(属性名必须是lr_scale)
            "params": [],
        }
    param_groups[group_name]["params"].append(param) # 将参数保存在字典的params属性中
    param_group_names[group_name]["params"].append(name)
```

将参数分组并设置`lr_scale`属性后，我们只需要将`param_group`作为参数传递给优化器，优化器会在更新对应参数时使用`lr * lr_scale`作为真实的学习率。

```
optimizer = flow.optim.Adam(list(param_groups.values()), lr=1e-3)
```

至此，我们完成了Graph模式下分层lr_scale的设置

### Eager模式下的lr_scale设置

在Eager模式下，除了以上的设置，我们还需要修改`LRscheduler`，继承`oneflow.optim.lr_scheduler._LRScheduler`模块，重写`update_lrs`函数，实现`group["lr"]`乘以`"lr_scale"`

```python
class LayerScaleLR(oneflow.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: flow.optim.Optimizer,
    ):
        super().__init__(optimizer)

    def update_lrs(self, lrs):
        self._last_lr = []
        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            if "lr_scale" in group:
                group["lr"] = lr * group["lr_scale"]
            else:
                group["lr"] = lr
            self._last_lr.append(lr)
```

随后我们只需要创建一个`LayerScaleLR`实例。

```python
optimizer = flow.optim.SGD(list(param_groups.values()), lr=1e-3)
lr_scheduler = LayerScaleLR(optimizer=optimizer)
```

在初始化时，会自动调用一次`update_lrs`函数，完成对不同`layer`学习率的设置。

