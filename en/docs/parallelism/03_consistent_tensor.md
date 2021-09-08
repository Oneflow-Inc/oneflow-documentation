# Consistent Tensor

## The mapping between consistent view and physical view

## Create Consistent tensor

To interactively experience consistent tensor on a two-GPU machine, you may launch python separately in two consoles in the following way.

!!! Note
    **click** the Terminal 0 or Terminal 1 label the check the commands/code in the two consoles




The above setting of environment variables is configuration for distributed computing. Please refer to the [extended reading](#_5) section at the end of this article for detailed explanation and launching distributed computing using some tools

### create consistent tensor directly

in the two consoles, separately import `oneflow` and create `x`

flow.palcement("cuda", {0:[0,1]}) appoints the range of consistent tensor in the cluster. - "cuda" means "on GPU". The second parameter of - placement is a dictionary. Its key is the index of machine, value is the index of graphic card. {0:[0,1]} means that consistent tensor is on the 0th, 1st graphics card of the 0th machine.

output:

Get local tensor from consistent tensor

calling to_local to check the local tensor on a device

convert local tensor to consistent tensor

User can create local tensor first, then use Tensor.to_consistent to convert local tensor to consistent tensor

In the following example, two local tensors of shape=(2, 5) are created on the two machines. Note that after calling to_consistent, the result consistent tensor has shape (4, 5)

This is because the chosen sbp=flow.sbp.split(0), two local tensor of shape (4, 5) needs to be concatenated on the 0th dimension and result in a (4, 5) consistent tensor.

Practice with SBP Signature

Data-parallelism

The following code is an example of data-parallelism of common distributed parallelism strategy

Observe that flow.matmul 