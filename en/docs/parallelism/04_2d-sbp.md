# 2D SBP

After reading the [Global View](./02_sbp.md) and [Global Tensor](./03_consistent_tensor.md), you may have learned about the basic concepts of SBP and SBP Signature, and can get started with related tasks. In fact, both these two documents refers to **1D SBP**.

Since you have known about 1D SBP, this document introduces 2D SBP, which can more flexibly deal with more complex distributed training scenarios.

## 2D Devices Array

We are already familiar with the placement configuration of 1D SBP. In the scenario of 1D SBP, configure the cluster through the [oneflow.placement](https://oneflow.readthedocs.io/en/master/placement.html#oneflow.placement) interface. For example, use the 0~3 GPU graphics in the cluster:

```python
>>> placement1 = flow.placement("cuda", ranks=[0, 1, 2, 3])
```

The above `"cuda"` specifies the device type, and `ranks=[0, 1, 2, 3]` specifies the computing devices in the cluster. In fact, `ranks` can be not only a one-dimensional int list, but also a multi-dimensional int array:

```python
>>> placement2 = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
```

When `ranks` is in the form of a one-dimensional list like `ranks=[0, 1, 2, 3]`, all devices in the cluster form a 1D device vector, which is where the 1D SBP name comes from.

When `ranks` is in the form of a multi-dimensional array, the devices in the cluster are grouped into a multi-dimensional array of devices. `ranks=[[0, 1], [2, 3]]` means that the four computing devices in the cluster are divided into a $2 \times 2$ device array.

## 2D SBP

When constructing a Global Tensor, we need to specify both `placement` and `SBP`. When the cluster in `placement` is a 2-dimensional device array, SBP must also correspond to it, being a `tuple` with a length of 2. The 0th and 1st elements in this `tuple` respectively describes the distribution of Global Tensor in the 0th and 1st dimensions of the device array.

For example, The following code configures a $2 \times 2$ device array, and sets the 2D SBP to `(broadcast, split(0))`.

```python
>>> a = flow.Tensor([[1,2],[3,4]])
>>> placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
>>> sbp = (flow.sbp.broadcast, flow.sbp.split(0))
>>> a_to_global = a.to_global(placement=placement, sbp=sbp)
```

It means that logically the data, over the entire device array, is `broadcast` in the 0th dimension ("viewed vertically"); `split(0)` in the 1st dimension ("viewed across").

See the following figure:

![](./imgs/2d-sbp.png)

In the above figure, the left side is the global data, and the right side is the data of each device on the device array. As you can see, from the perspective of the 0th dimension, they are all in `broadcast` relations:

- The data in (group0, device0) and (group1, device0) are consistent, and they are in `broadcast` relations to each other
- The data in (group0, device1) and (group1, device1) are consistent, and they are in `broadcast` relations to each other

From the perspective of the 1st dimension, they are all in `split(0)` relations:

- (group0, device0) and (group0, device1) are in `split(0)` relations to each other
- (group1, device0) and (group1, device1) are in `split(0)` relations to each other

It may be difficult to directly understand the correspondence between logical data and physical data in the final device array. When thinking about 2D SBP, you can imagine an intermediate state (gray part in the above figure) there. Take `(broadcast, split(0))` as an example:

- First, the original logical tensor is broadcast to 2 groups through `broadcast`, and the intermediate state is obtained
- On the basis of the intermediate state, `split(0)` is continued to be done on the groups to get the status of each physical tensor in the final device array

## 2D SBP Signature

1D SBP has the concept of SBP signature, similarly, the operator also has 2D SBP signature. Based on mastering the concept of 1D SBP and its signature concept, 2D SBP signature is very simple and you only need to follow one principle:

- Independently derive in the respective dimensions

Let's take matrix multiplication as an example. First, let's review the case of 1D SBP. Suppose that $x \times w = y$ can have the following SBP Signature:

$$ broadcast \times split(1) = split(1) $$

and

$$ split(0) \times broadcast = split(0) $$

Now, suppose we set the 2D SBP for $x$ to $(broadcast, split(0))$ and set the 2D SBP for $w$ to $(split(0), broadcast)$, then in the context of the 2D SBP, operate $x \times w = y$ to obtain the SBP attribute for $y$ is $(split(1), split(0))$.

That is to say, the following 2D SBPs constitute the 2D SBP Signature of matrix multiplication:

$$ (broadcast, split(0)) \times (split(1), broadcast) =  (split(1), split(0)) $$
