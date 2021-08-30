# Compatible Interface

This topic is only reserved for compatibility with the old interface. If you haven't used 'v0.4.0' or earlier versions, please **ignore all the contents of this topic directly** .

By version 0.4.0 and earlier, the interface of OneFlow was non object-oriented. Now OneFlow provides object-oriented interfaces.

In order to take care of the code using the old interface, OneFlow moves the old interface to `oneflow.compatible.single_client`.

That code only need to replace the import statements:

```python
import oneflow as flow
import oneflow.typing as tp
```

with:

```python
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
```
