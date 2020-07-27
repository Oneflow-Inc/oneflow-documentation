
## BERT (Bidirectional Encoder Representations from Transformer) is a new type of pre-training model in the NLP field. In this case, based on the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, the OneFlow version of the BERT model is implemented.
BERT(Bidirectional Encoder Representations from Transformers)是NLP领域的一种新型预训练模型。本案例中，基于论文[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)实现了BERT模型的OneFlow版本。本案例中，基于论文[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)实现了BERT模型的OneFlow版本。

### Model Architecture
| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:-----------------:|:--------------------:|:-------------------:|:---------------------------:|:-----------------------:|:--------------:|
| BERTBASE  |    12 encoder     |         768          |         12          |          4 x  768           |           512           |      110M      |

BERT is often divided into two steps in practical applications

* First, pre-train to get Bert model

* Then, in order to meet the needs of downstream applications, on the basis of the obtained Bert language model, an additional layer of network is added and fine-tuned to obtain downstream applications.


## Quick Start
### Get the dataset
We provide [the OFRecord dataset and the related data files](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/bert_squad_dataset.zip) that have completed BERT pre-training and SQuAD fine-tuning, which can be downloaded and decompressed by the following command:

```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/bert_squad_dataset.zip
unzip bert_squad_dataset.zip
```
The decompressed file directory list is as follows:

* bert_config.json, vocab.txt: files needed to make prediction json files, from [google bert](https://github.com/google-research/bert)

* dev-v1.1/, dev-v1.1.json: SQuAD test set, used for scoring

* part-0: Pre-training set samples (40 samples)

* train-v1.1: SQuAD training set, which has been converted to ofrecord data set format

The files above will be used in the pre-training tasks and SQuAD fine-tuning below.

### Train BERT model
First, clone `OneFlow-Benchmark` repository.

```bash
git clone https://github.com/Oneflow-Inc/OneFlow-Benchmark.git
cd OneFlow-Benchmark/LanguageModeling/BERT/
```

Then, use our pre-trained pre-trained model and a small sample set to start BERT pre-training to see the effect with the following command:
```bash
python ./run_pretraining.py\
    --gpu_num_per_node=1 \
    --learning_rate=3e-5 \
    --batch_size_per_device=1 \
    --iter_num=3 \
    --loss_print_every_n_iter=50 \
    --seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_hidden_layers=12 \
    --num_attention_heads=12 \
    --max_position_embeddings=512 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --attention_probs_dropout_prob=0.0 \
    --hidden_dropout_prob=0.0 \
    --hidden_size_per_head=64 \
    --use_boxing_v2=True \
    --data_dir=./dataset/ \
    --data_part_num=1 \
    --log_dir=./bert_regresssioin_test/of \
    --loss_print_every_n_iter=5 \
    --model_save_dir=./bert_regresssioin_test/of \
    --warmup_batches 831 \
    --save_last_snapshot True 
```
The following outputs are expected:
```text
==================================================================
Running bert: num_gpu_per_node = 1, num_nodes = 1. ==================================================================
gpu_num_per_node = 1
node_num = 1
node_list = None
learning_rate = 3e-05
weight_decay_rate = 0.01
batch_size_per_device = 1
iter_num = 20
warmup_batches = 831
log_every_n_iter = 1
data_dir = ./dataset/
data_part_num = 1
use_fp16 = None
use_boxing_v2 = True
loss_print_every_n_iter = 5
model_save_every_n_iter = 10000
model_save_dir = ./bert_regresssioin_test/of
save_last_snapshot = True
model_load_dir = None
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
Time stamp: 2020-07-06-19:09:29
I0706 19:09:29.605840639   34801 ev_epoll_linux.c:82]        Use of signals is disabled. Epoll engine will not be used
Init model on demand
iter 4, total_loss: 11.032, mlm_loss: 10.281, nsp_loss: 0.751, speed: 33.086(sec/batch), 0.151(sentences/sec)
iter 9, total_loss: 11.548, mlm_loss: 10.584, nsp_loss: 0.965, speed: 0.861(sec/batch), 5.806(sentences/sec)
iter 14, total_loss: 10.697, mlm_loss: 10.249, nsp_loss: 0.448, speed: 0.915(sec/batch), 5.463(sentences/sec)
iter 19, total_loss: 10.685, mlm_loss: 10.266, nsp_loss: 0.419, speed: 1.087(sec/batch), 4.602(sentences/sec)
Saving model to ./bert_regresssioin_test/of/last_snapshot. ------------------------------------------------------------------
average speed: 0.556(sentences/sec)
------------------------------------------------------------------
```

## Detailed Explanation
### Explanation of the script
|         **Scripts**          |                                                                                                                      **Explanation**                                                                                                                       | **Belong to** |
|:----------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|
|     pretrain.py、bert.py      |                                                                                                                     Defines BERT model                                                                                                                     |     BERT      |
|      run_pretraining.py      | Starts the user script for BERT training. The user can configure the training environment and hyperparameters of BERT training through command line parameters. The specific functions of each parameter will be explained in **Script Parameters** below. |     BERT      |
|           squad.py           |                                                                                                                    Defines SQuAD model                                                                                                                     |     SQuAD     |
|         run_squad.py         |                                                                                                              Starts the SQuAD model training.                                                                                                              |     SQuAD     |
|    run_squad_predict.py    |                                                                                                        Use the trained SQuAD model for prediction.                                                                                                         |     SQuAD     |
|         npy2json.py          |                                                                                           Converts the OneFlow prediction results to the prediction json format.                                                                                           |     SQuAD     |
| convert_tf_ckpt_to_of.py |                                                                                                        Converts TensorFlow model to OneFlow model.                                                                                                         |  BERT/SQuAD   |



### Script Parameters
`run_pretraining.py` uses command arguments to configure the training environment including hyperparameters. You can check with `run_pretraining.py --help`. The detailed explanation for these parameters is as follows:

* gpu_num_per_node: number of GPUs in each node. OneFlow requires the same number of GPUs in each node.

* node_num: number of nodes, that is, the number of hosts during distributed training.

* node_list: node list, if the number of nodes is greater than 1, you need to specify the node list through node_list, the node list is in the form of a string, separated by commas, such as `--node_num=2 --node_list="192.168.1.12,192.168. 1.14`"

* learning_rate： Learning rate

* weight_decay_rate: Set the weight decay rate

* batch_size_per_device: batch size on each device during distributed training

* iter_num ITER_NUM: total number of training iterations

* warmup_batches: the number of warm up batches, the default value is 10000

* data_dir: the directory of the OFRecord dataset

* data_part_num: The number of data files in the OFRecord dataset directory

* use_fp16: whether or not to use float16

* use_boxing_v2: whether or not to use boxing v2

* loss_print_every_n_iter: print training information (loss information) every n iterations during training

* model_save_every_n_iter: save model every n iterations during training

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

### 使用完整的Wikipedia + BookCorpus数据集
如果需要无到有进行BERT的pretrain训练，则需要使用较大的训练集。

如果感兴趣，可以通过[google-research BERT](https://github.com/google-research/bert)的页面，下载tfrecord格式的数据集。再根据[加载与准备OFRecord数据集](../extended_topics/how_to_make_ofdataset.md)中的方法，将TFRecord数据转为OFRecord数据集使用。再根据[加载与准备OFRecord数据集](../extended_topics/how_to_make_ofdataset.md)中的方法，将TFRecord数据转为OFRecord数据集使用。

### 将Tensorflow的BERT模型转为OneFlow模型格式
如果想直接使用已经训练好的pretrained模型做fine-tune任务（如以下将展示的SQuAD），可以考虑直接从[google-research BERT](https://github.com/google-research/bert)页面下载已经训练好的BERT模型。

再利用我们提供的`convert_tf_ckpt_to_of.py`脚本，将其转为OneFlow模型格式。转换过程如下：转换过程如下：

首先，下载并解压某个版本的BERT模型，如`uncased_L-12_H-768_A-12`。
```shell
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d uncased_L-12_H-768_A-12
```

然后，运行以下命令：
```shell
cd uncased_L-12_H-768_A-12/
cat > checkpoint <<ONEFLOW
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
ONEFLOW
```

该命令将在解压目录下创建一个`checkpoint`文件，并写入以下内容：
```
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
```

此时，已经准备好待转化的tensorflow模型目录，整个模型目录的结构如下：
```shell
uncased_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── checkpoint
└── vocab.txt
```

我们接着使用`convert_tf_ckpt_to_of.py`将tensorflow模型转为OneFlow模型：
```bash
python convert_tf_ckpt_to_of.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12 \
  --of_dump_path ./uncased_L-12_H-768_A-12-oneflow
```
以上命令，将转化好的OneFlow格式的模型保存在`./uncased_L-12_H-768_A-12-oneflow`目录下，供后续微调训练(如SQuAD)使用。

## 微调：SQuAD问答任务
### 将pretrained模型修改为SQuAD模型
我们只需要在BERT的backbone基础上，加上一层`output`层，并修改loss的表达式即可，完整的代码可以查看`squad.py`脚本，以下是几处关键修改：
```python
def SQuADTrain():
    #... backbone = bert_util.BertBackbone()

    #在BERT的基础上加上一个全连接层
    with flow.name_scope("cls-squad"):
        final_hidden = backbone.sequence_output()
        final_hidden_matrix = flow.reshape(final_hidden, [-1, hidden_size])
        logits = bert_util._FullyConnected(
                    final_hidden_matrix,
                    hidden_size,
                    units=2,
                    weight_initializer=bert_util.CreateInitializer(initializer_range),
                    name='output')
        logits = flow.reshape(logits, [-1, seq_length, 2])

        start_logits = flow.slice(logits, [None, None, 0], [None, None, 1])
        end_logits = flow.slice(logits, [None, None, 1], [None, None, 1])

    #重新定义SQuAD任务的loss
        start_loss = _ComputeLoss(start_logits, start_positions_blob, seq_length)
        end_loss = _ComputeLoss(end_logits, end_positions_blob, seq_length)

        total_loss = 0.5*(start_loss + end_loss)

    return total_loss
```

为了得到一个初始化的squad模型，我们通过以下脚本启动squad训练，并保存模型。

```shell
python ./run_squad.py\
    --gpu_num_per_node=1\
    --learning_rate=3e-5\
    --batch_size_per_device=2\
    --iter_num=50\
    --loss_print_every_n_iter=50\
    --seq_length=384\
    --max_predictions_per_seq=20\
    --num_hidden_layers=12\
    --num_attention_heads=12\
    --max_position_embeddings=512\
    --type_vocab_size=2\
    --vocab_size=30522\
    --attention_probs_dropout_prob=0.0\
    --hidden_dropout_prob=0.0\
    --hidden_size_per_head=64\
    --use_boxing_v2=True\
    --data_dir=./dataset/train-v1.1\
    --data_part_num=1\
    --log_dir=./bert_regresssioin_test/of\
    --model_save_dir=./bert_regresssioin_test/of\
    --warmup_batches 831\
    --save_last_snapshot True
```
完成训练后，在`./bert_regresssioin_test/of/last_snapshot`中保存有初始化的SQuAD模型，我们将其与训练好的SQuAD合并后，进行微调（fine-tune）训练。

### 合并pretrained模型为SQuAD模型
SQuAD模型是在pretrained模型基础上的扩充，我们需要参照[模型的加载与保存](../basics_topics/model_load_save.md)中的“模型部分初始化和部分导入”方法，将训练好的BERT pretrained模型与初始化的SQuAD模型合并。

```shell
cp -R ./bert_regresssioin_test/of/last_snapshot ./squadModel
cp -R --remove-destination ./dataset/uncased_L-12_H-768_A-12_oneflow/* ./squadModel/
```

### OneFlow预训练模型的训练次数问题
OneFlow生成的模型目录中，会有一个名为`System-Train-TrainStep-xxx`的子目录(xxx为任务函数的函数名)，该子目录下的out文件中，保存有训练总迭代数，并且这个迭代数会用于动态调节训练过程的`learning rate`。

为了防止保存的迭代数影响到微调的训练，应该将out文件中的二进制数据清零：
```shell
cd System-Train-TrainStep-xxx
xxd -r > out <<ONEFLOW
00000000: 0000 0000 0000 0000
ONEFLOW
```

如果你使用的是由TensorFlow转过来的预训练模型，则可以省去这个步骤。

### 开始SQuAD训练
通过`run_suqad.py`脚本，开始训练SQuAD模型，主要配置如下：

* 使用以上合并得到的SQuAD模型`./squadModel`

* 采用SQuAD v1.1作为训练集

* epoch = 3 (`iternum = 88641*3/(4*8) = 8310`)

* learning rate = 3e-5

```shell
python ./run_squad.py\
    --gpu_num_per_node=4\
    --learning_rate=3e-5\
    --batch_size_per_device=8\
    --iter_num=8310\
    --loss_print_every_n_iter=50\
    --seq_length=384\
    --max_predictions_per_seq=20\
    --num_hidden_layers=12\
    --num_attention_heads=12\
    --max_position_embeddings=512\
    --type_vocab_size=2\
    --vocab_size=30522\
    --attention_probs_dropout_prob=0.0\
    --hidden_dropout_prob=0.0\
    --hidden_size_per_head=64\
    --use_boxing_v2=True\
    --data_dir=./dataset/train-v1.1\
    --data_part_num=8\
    --log_dir=./bert_regresssioin_test/of\
    --model_save_dir=./bert_regresssioin_test/of\
    --warmup_batches 831\
    --save_last_snapshot True\
    --model_load_dir=./squadModel
```

### 预测及打分
生成为了生成[Preidiction File](https://rajpurkar.github.io/SQuAD-explorer/)格式的json文件，我们先将预测结果保存为npy文件，再使用[google BERT的run_squad.py](https://github.com/google-research/bert/blob/master/run_squad.py)中的`write_predictions`函数，转化为json格式。

利用`run_squad_predict.py`生成`all_results.npy`文件：
```bash
python run_squad_predict.py \
  --gpu_num_per_node=1 \
  --batch_size_per_device=4 \
  --iter_num=2709 \
  --seq_length=384 \
  --max_predictions_per_seq=20 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.0 \
  --hidden_dropout_prob=0.0 \
  --hidden_size_per_head=64 \
  --use_boxing_v2=True \
  --data_part_num=1 \
  --data_dir=./dataset/dev-v1.1 \
  --log_dir=./bert_regresssioin_test/of \
  --model_load_dir=path/to/squadModel \
  --warmup_batches 831
```
注意将以上`model_load_dir`修改为 **训练好的** squadModel。

得到`all_results.npy`文件后，在[google bert](https://github.com/google-research/bert/)仓库目录下（注意该仓库的tensorflow版本为 **tensorflow v1** ），运行我们提供的`npy2json.py`(由google bert中的run_squand.py修改得来)：
```shell
python npy2json.py\
  --vocab_file=./dataset/vocab.txt \
  --bert_config_file=./dataset/bert_config.json \
  --do_train=False \
  --do_predict=True \
  --all_results_file=./all_results.npy \
  --predict_file=./dataset/dev-v1.1.json \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=./squad_base/
```

注意将`all_results_file`修改为上一步得到的`all_results.npy`的路径。

最终，得到`predictions.json`文件，可以使用[evaluate-v1.1.py](https://rajpurkar.github.io/SQuAD-explorer/)进行打分。

```bash
python evaluate-v1.1.py \
./dataset/dev-v1.1.json \
path/to/squad_base/predictions.json 
```

## 分布式训练
如之前介绍脚本参数时描述：进行分布式训练，只需要在启动训练脚本式加入`node_num`选项指定主机数目及 `node_list`选项即可：

```bash
python run_squad_predict.py \
  --gpu_num_per_node=1 \
  --batch_size_per_device=4 \
  --iter_num=2709 \
  --seq_length=384 \
  --max_predictions_per_seq=20 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.0 \
  --hidden_dropout_prob=0.0 \
  --hidden_size_per_head=64 \
  --use_boxing_v2=True \
  --data_part_num=1 \
  --data_dir=./dataset/dev-v1.1 \
  --log_dir=./bert_regresssioin_test/of \
  --model_load_dir=path/to/squadModel \
  --warmup_batches 831 \
  --node_num=2 \
  --node_list="192.168.1.12,192.168.1.14"
```