import numpy as np
import oneflow as flow
import oneflow.typing as oft

BATCH_SIZE = 100

def mlp(data):
  initializer = flow.truncated_normal(0.1)
  reshape = flow.reshape(data, [data.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer)

def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.default_distribute_strategy(flow.scope.consistent_view())
  config.train.model_update_conf({"naive_conf": {}})
  return config

def config_distributed():
    print("distributed config")
    #每个节点的gpu使用数目
    flow.config.gpu_device_num(1)
    #通信端口
    flow.env.ctrl_port(9988)

    #节点配置
    nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
    flow.env.machine(nodes)

@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)) -> oft.Numpy:
  logits = mlp(images)
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  config_distributed()
  flow.config.enable_debug_mode(True)
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
  for epoch in range(1):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels)
      if i % 20 == 0: print(loss.mean())
  #check_point.save('./mlp_models_1') # need remove the existed folder