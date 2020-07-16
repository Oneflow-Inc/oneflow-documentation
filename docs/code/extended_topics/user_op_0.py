import oneflow as flow
import numpy as np
# 默认配置
flow.config.gpu_device_num(1)
# 加载模块
flow.config.load_library("relu.so")
# python op wrapper function
def relu(input_blob, op_name):
  return flow.user_op_builder(op_name).Op("Relu").Input("in", [input_blob]).Build().RemoteBlobList()[0]

# 定义你的Job的配置
my_func_config = flow.FunctionConfig()                                                                 
my_func_config.default_distribute_strategy(flow.distribute.consistent_strategy())                      
my_func_config.default_data_type(flow.float) 
# 网络代码
@flow.function(my_func_config)
def MyJob(x = flow.FixedTensorDef((5,))):
  return relu(x, "my_relu_op_name")

# 执行
input_data = [-2,-1,0,1,2]
output_data = MyJob(np.array(input_data, dtype=np.float32)).get().ndarray()
print(output_data)

# 期望执行结果
[0. 0. 0. 1. 2.]
