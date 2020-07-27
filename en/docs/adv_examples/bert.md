
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

* model_save_dir: the path where the model will be saved

* save_last_snapshot: specify the path to save the model after the last round of training is completed

* model_load_dir: specify the model loading path

* log_dir LOG_DIR: specify the log path

* seq_length: specify the length of the BERT sentence, the default value is 512

* max_predictions_per_seq: the default value is 80

* num_hidden_layers: the number of hidden layers, the default value is 24

* num_attention_heads: the number of Attention heads, the default value is 16

* max_position_embeddings：

* type_vocab_size

* vocab_size

* attention_probs_dropout_prob

* hidden_dropout_prob

* hidden_size_per_head

### Use the complete Wikipedia + BookCorpus data set
If you need to perform Bert pre-train training from scratch, you need to use a larger training set.

If you are interested, you can download the data set in tfrecord format through the page of [google-research BERT](https://github.com/google-research/bert). Then according to [ the method in loading and preparing OFRecord data set](../extended_topics/how_to_make_ofdataset. md), convert TFRecord data to OFRecord data set for use.再根据[加载与准备OFRecord数据集](../extended_topics/how_to_make_ofdataset.md)中的方法，将TFRecord数据转为OFRecord数据集使用。

### Convert Tensorflow's BERT model to OneFlow model format
If you want to directly use the trained pretrained model for fine-tune tasks (such as SQuAD shown below), you can consider downloading the trained BERT model directly from the [google-research BERT](https://github.com/google-research/bert) page.

Then use the `convert_tf_ckpt_to_of.py` script we provided to convert it to the OneFlow model format.The conversion process is as follows:

First, download and decompress a certain version of the BERT model, such as `uncased_L-12_H-768_A-12`.
```shell
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d uncased_L-12_H-768_A-12
```

Then, run the following command:
```shell
cd uncased_L-12_H-768_A-12/
cat > checkpoint <<ONEFLOW
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
ONEFLOW
```

This command will create a `checkpoint` file in the decompressed directory and write the following content:
```
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
```

At this point, the tensorflow model directory to be converted is ready. The structure of the entire model directory is as follows
```shell
uncased_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── checkpoint
└── vocab.txt
```

We then use `convert_tf_ckpt_to_of.py` to convert the tensorflow model to the OneFlow model:
```bash
python convert_tf_ckpt_to_of.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12 \
  --of_dump_path ./uncased_L-12_H-768_A-12-oneflow
```
The above command saves the converted OneFlow format model in the `./uncased_L-12_H-768_A-12-oneflow` directory for subsequent fine-tuning training (such as SQuAD).

## Fine-tuning: SQuAD question and answer task
### Modify the pretrained model to SQuAD model
We only need to add a layer of `output` on the basis of BERT's backbone, and modify the expression of loss. The complete code can be viewed in the `squad.py` script. Here are a few key changes:
```python
def SQuADTrain():
    #... backbone = bert_util.BertBackbone()

    #Add a full-connected layer to BERT
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

    #Redefine the loss of SQuAD task
        start_loss = _ComputeLoss(start_logits, start_positions_blob, seq_length)
        end_loss = _ComputeLoss(end_logits, end_positions_blob, seq_length)

        total_loss = 0.5*(start_loss + end_loss)

    return total_loss
```

In order to get an initialized squad model, we start squad training through the following script and save the model.

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
After the training is completed, the initialized SQuAD model is saved in `./bert_regresssioin_test/of/last_snapshot`, we merge it with the trained SQuAD, and perform fine-tune training.

### Combine pretrained models into SQuAD models
The SQuAD model is an expansion on the basis of the pretrained model. We need to refer to the "model partial initialization and partial import" method in [Model Loading and Saving](../basics_topics/model_load_save.md), to combine the trained BERT pretrained model with the initialized SQuAD model.

```shell
cp -R ./bert_regresssioin_test/of/last_snapshot ./squadModel
cp -R --remove-destination ./dataset/uncased_L-12_H-768_A-12_oneflow/* ./squadModel/
```

### The number of training times of the OneFlow pre-training model
In the model directory generated by OneFlow, there will be a subdirectory named `System-Train-TrainStep-xxx` (xxx is the function name of the task function). The out file under this subdirectory contains The total number of training iterations, and this number of iterations will be used to dynamically adjust the `learning rate` of the training process.

In order to prevent the saved iterations from affecting the fine-tuning training, the binary data in the out file should be cleared:
```shell
cd System-Train-TrainStep-xxx
xxd -r > out <<ONEFLOW
00000000: 0000 0000 0000 0000
ONEFLOW
```

If you are using a pre-trained model converted from TensorFlow, you can omit this step.

### Start the training of SQuAD
Start training the SQuAD model through the `run_suqad.py` script, the main configuration is as follows:

* Use the SQuAD model obtained by the above merge `./squadModel`

* Use SQuAD v1.1 as the training set

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

### Forecast and grading
To generate a json file in the format of [Preidiction File](https://rajpurkar.github.io/SQuAD-explorer/), we first save the prediction result as an npy file, and then use [write_predictions](https://github.com/google-research/bert/blob/master/run_squad.py) in `google BERT’s run_squad.py` Function, converted to json format.

Use `run_squad_predict.py` to generate `all_results.npy` file:
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
Note that you need to modify the above `model_load_dir` to **trained** squadModel.

After obtaining the `all_results.npy` file, in the [google bert](https://github.com/google-research/bert/) warehouse directory (note that the tensorflow version of the warehouse is **tensorflow v1**), run the provided `npy2json.py` (modified from run_squand.py in google bert):
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

Note that you need to change `all_results_file` to the path of `all_results.npy` obtained in the previous step.

Finally, you will get the `predictions.json` file, which can be scored using [evaluate-v1.1.py](https://rajpurkar.github.io/SQuAD-explorer/).

```bash
python evaluate-v1.1.py \
./dataset/dev-v1.1.json \
path/to/squad_base/predictions.json 
```

## Distributed Training
As described in the previous explanation of script parameters: in distributed training, you only need to add the `node_num` option to specify the number of hosts and `node_list` option when starting the training script:

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