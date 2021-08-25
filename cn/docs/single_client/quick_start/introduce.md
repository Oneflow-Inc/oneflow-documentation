# 原接口兼容

本专题仅仅是为兼容原接口而保留。如果您未使用过 `v0.4.0` 及之前的版本，请 **直接忽略掉本专题的所有内容**。

在 OneFlow v0.4.0 版本及以前，OneFlow 的接口是非面向对象的。现有版本的动态图、静态图模式均提供了面向对象接口。

为了照顾老接口的用户，OneFlow 将老接口移动至 `oneflow.compatible.single_client`。
有历史遗留代码的用户，只需要将原代码的包导入：

```python
import oneflow as flow
import oneflow.typing as tp
```

改为新的导入方式即可：

```python
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
```
