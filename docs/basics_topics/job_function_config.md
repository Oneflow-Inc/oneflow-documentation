
OneFlow在算力管理上有巨大优势，为了更大限度地发挥OneFlow框架优势，OneFlow中使用 **任务函数** (job function)关联用户的业务逻辑与OneFlow管理的计算资源。
我们可以将自己的业务逻辑（网络模型构建、参数交互等）封装为一个普通python函数，然后使用OneFlow中的修饰符将该函数转变为任务函数。
虽然封装业务逻辑的 **普通python函数** ，与转变后的 **任务函数** 其实是两个概念，但它们两者的转化过程对用户是透明的，因此在本文我们不做严格区分，都统称为 **任务函数** 。

## oneflow.config配置对象

我们将使用`oneflow.function`修饰符修饰任务函数，它需要一个配置对象，由`oneflow.function_config`创建。
config对象中的成员方法对应了模型训练过程中的软硬件配置。
以下是是一个基本的训练配置，设置了learning rate，模型更新方案：

```python
def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config
```

以下是一个基本的校验配置，去掉了训练过程才需要的配置：

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config
```

## oneflow.function修饰符

设置好config对象后，将config对象传递给修饰符`oneflow.function`就可以将普通函数转化为训练任务。

```python
@flow.function(get_train_config())
def train_job(images_blob=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float)):
  #...
```

## config对象中的配置参数

config对象中的与软硬件配置有关的参数有：

```python
all_reduce_fp16
all_reduce_group_min_mbyte
all_reduce_group_num
all_reduce_group_size_warmup
all_reduce_lazy_ratio
allow_cpu_return_op
concurrency_width
cudnn_buf_limit_mbyte
cudnn_conv_enable_pseudo_half
cudnn_conv_enable_true_half
cudnn_conv_force_bwd_data_algo
cudnn_conv_force_bwd_filter_algo
cudnn_conv_force_fwd_algo
cudnn_conv_heuristic_search_algo
cudnn_conv_use_deterministic_algo_only
default_data_type
default_distribute_strategy
default_initializer_conf
default_placement_scope
disable_all_reduce_sequence
enable_all_reduce_group
enable_auto_mixed_precision
enable_cudnn
enable_cudnn_conv_pseudo_half
enable_float_compute_for_half_gemm
enable_inplace
enable_inplace_in_reduce_struct
enable_keep_header_only
enable_nccl
enable_non_distributed_optimizer
enable_reused_mem
enable_true_half_config_when_conv
exp_run_conf
function_desc
indexed_slices_optimizer_conf
non_distributed_optimizer_group_size_mbyte
prune_parallel_cast_ops
static_mem_alloc_algo_white_list
static_mem_alloc_policy_white_list
tensorrt
train
use_boxing_v2
use_memory_allocation_algorithm_v2
use_nccl_inter_node_communication
use_tensorrt
use_xla_jit
```

其中train成员，专门用于配置与 **训练模型** 有关的参数，train的成员有：

```python
function_desc
loss_scale_factor
model_update_conf
primary_lr
secondary_lr
```

