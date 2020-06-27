
## 模型概述
BERT(Bidirectional Encoder Representations from Transformers)是NLP领域的一种新型预训练模型。本案例中，基于论文[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)实现了BERT模型的OneFlow版本。

### 模型架构
| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTBASE |12 encoder| 768| 12|4 x  768|512|110M|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|

BERT在实际应用中往往分为两步：

* 首先，预训练得到BERT语言模型；

* 然后，为满足下游应用，在得到的BERT语言模型的基础上，多加一层网络，并进行微调，得到下游应用。


## 快速开始
### 获取ofrecord数据集
我们提供了已经制作好的[SQuAD](https://link_toSquAD)(Stanford Question Answering Dataset)OFRecord数据集，可以通过以下命令下载并解压：

```bash
wget https://link_toSquAD/squad.zip
unzip squad.zip
```

该OFRecord数据集的路径将在训练BERT模型中使用。

### 训练BERT模型
首先，克隆`OneFlow-Benchmark`仓库。

```bash
git clone https://github.com/Oneflow-Inc/OneFlow-Benchmark.git
cd OneFlow-Benchmark/bert_benchmark/
```

然后，通过以下命令，开始BERT预训练：
```bash

```
我们将获得类似以下输出：
```text
==================================================================
Running bert: num_gpu_per_node = 4, num_nodes = 1.
==================================================================
gpu_num_per_node = 4
node_num = 1
node_list = None
learning_rate = 0.0001
weight_decay_rate = 0.01
batch_size_per_device = 24
iter_num = 100000
warmup_batches = 10000
log_every_n_iter = 1
data_dir = /dataset/bert/of_wiki_seq_len_128
data_part_num = 64
use_fp16 = None
use_boxing_v2 = True
loss_print_every_n_iter = 20
model_save_every_n_iter = 100000
model_save_dir = ./bert_regresssioin_test/of
save_last_snapshot = False
model_load_dir = 
log_dir = ./bert_regresssioin_test/of
seq_length = 128
max_predictions_per_seq = 20
num_hidden_layers = 12
num_attention_heads = 12
max_position_embeddings = 512
type_vocab_size = 2
vocab_size = 30522
attention_probs_dropout_prob = 0.0
hidden_dropout_prob = 0.0
hidden_size_per_head = 64
------------------------------------------------------------------
Time stamp: 2020-06-27-17:26:48
I0627 17:26:48.099260502   30032 ev_epoll_linux.c:82]        Use of signals is disabled. Epoll engine will not be used
```
等待OneFlow框架配置完成后，训练开始：
```text
Init model on demand
iter 19, total_loss: 10.987, mlm_loss: 10.291, nsp_loss: 0.695, speed: 47.756(sec/batch), 40.204(sentences/sec)
iter 39, total_loss: 10.648, mlm_loss: 9.978, nsp_loss: 0.670, speed: 9.822(sec/batch), 195.487(sentences/sec)
iter 59, total_loss: 10.350, mlm_loss: 9.659, nsp_loss: 0.691, speed: 9.695(sec/batch), 198.035(sentences/sec)
iter 79, total_loss: 10.203, mlm_loss: 9.525, nsp_loss: 0.678, speed: 9.910(sec/batch), 193.734(sentences/sec)
...
```

## 详细说明
### 脚本说明

* pretrain.py、bert.py：定义了BERT网络模型；

* benchmark_util.py：包含了一些打印训练情况的辅助类；

* run_pretraining.py：启动BERT训练的用户脚本，用户通过命令行参数进行BERT训练的训练环境及超参配置，各个参数的具体作用将在下文 **脚本参数** 中说明。

### 脚本参数
`run_pretraining.py`通过命令行参数配置包括超参在内的训练环境，可以通过
`run_pretraining.py --help`查看，以下是这些参数作用的具体说明：

* gpu_num_per_node： 每个节点上GPU的数目，OneFlow要求每个节点的GPU数目必须一致

* node_num： 节点数目，即分布式训练时的主机数目

* node_list： 节点列表，如果节点数大于1，则需要通过node_list指定节点列表，节点列表为字符串形式，采用逗号分隔，如`--node_num=2 --node_list="192.168.1.12,192.168.1.14`"

* learning_rate： Learning rate

* weight_decay_rate：设置权重衰减率

* batch_size_per_device： 分布式训练时每个设备上的batch大小

* iter_num ITER_NUM： 训练的总轮数

* warmup_batches： 预热轮数，默认值为10000

* data_dir： OFRecord数据集的路径

* data_part_num：OFRecord数据集目录下的数据文件数目

* use_fp16： 是否使用fp16

* use_boxing_v2： 是否使用boxing v2

* loss_print_every_n_iter：训练中每隔多少轮打印一次训练信息（loss信息）

* model_save_every_n_iter： 训练中每隔多少轮保存一次模型

* model_save_dir： 模型存储路径

* save_last_snapshot：指定最后一轮训练完成后，模型保存路径

* model_load_dir：指定模型加载路径

* log_dir LOG_DIR：指定日志路径

* seq_length： 指定BERT句子长度，默认值为512

* max_predictions_per_seq： 默认值为80

* num_hidden_layers：隐藏层数目，默认值为24

* num_attention_heads： Attention头数目，默认值为16

* max_position_embeddings：

* type_vocab_size 

* vocab_size 

* attention_probs_dropout_prob 

* hidden_dropout_prob

* hidden_size_per_head 
```


## 数据集的制作


## 性能表现

## 下游应用：SQuAD问答任务