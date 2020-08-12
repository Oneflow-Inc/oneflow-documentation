
## Summary
BERT(Bidirectional Encoder Representations from Transformers) is a technique for NLP. In our case, we implement BERT based on the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) using OneFlow.

### Model
| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:-----------------:|:--------------------:|:-------------------:|:---------------------------:|:-----------------------:|:--------------:|
| BERTBASE  |    12 encoder     |         768          |         12          |          4 x  768           |           512           |      110M      |

There are commonly two steps in BERT:

* First, BERT pretrained model is obtained by pre-training;

* Then, on the basis of the obtained pretrained model, an additional layer of network is added and finetuned to get the downstream application.


## Quickstart
### Get dataset
We provide [OFRecord dataset and relevant other files](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/bert_squad_dataset.zip), you can get and unzip it by running commands below:

```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/bert_squad_dataset.zip
unzip bert_squad_dataset.zip
```
The list of files is as follows:

* bert_config.json、vocab.txt：Files needed to generate "prediction json" file from [google bert](https://github.com/google-research/bert)

* dev-v1.1/, dev-v1.1.json：SQuAD test set for evaluation

* part-0：pre-trained training set contains 40 samples

* train-v1.1：SQuAD training set that has been coverted to OFRecords

The above files will be used in the following pretraining tasks and squad finetune.

### BERT pretrained
Firstly, clone the `OneFlow-Benchmark`:

```bash
git clone https://github.com/Oneflow-Inc/OneFlow-Benchmark.git
cd OneFlow-Benchmark/LanguageModeling/BERT/
```

Then, with the following command, we can use our pretraining model and small sample set to start the BERT pre-training.
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

We will see the output similar to the following:
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

## Detailed description
### Scripts
| **Files** | **Description** | **Belongs to**|
|:---------:|:----------:|:----------:|
|pretrain.py、bert.py| Define the BERT model |BERT|
|run_pretraining.py|Start BERT training. The user can configure the training environment and parameters of the BERT training through the command line parameters. The specific meanings of each option will be described in the **script options** below.| BERT|
|squad.py|define SQuAD network|SQuAD|
|run_squad.py|Run the SQuAD training|SQuAD|
|run_squad_predict.py|Run the trained SQuAD model to predict.|SQuAD|
|npy2json.py|Script required to overt OneFlow's prediction results to json.|SQuAD|
|convert_tf_ckpt_to_of.py|Convert model from TensorFlow to OneFlow|BERT/SQuAD|



### Options
The script `run_pretraining.py` runs the pretraining and configured by command line options. You can run `run_pretraining.py --help` to see the options. The following is a detailed description of each option：

* gpu_num_per_node: count of devices on each node which must be consistent on each machine

* node_num: count of nodes, that is, the count of hosts in distributed system

* node_list: list of nodes. When thec count of nodes is more than one, we should spcifiy list of nodes by node_list. It's a string seperated by commans like `--node_num=2 --node_list="192.168.1.12,192.168.1.14"`

* learning_rate: learning rate

* weight_decay_rate: decay rate of weight

* batch_size_per_device: batch size on each device

* iter_num ITER_NUM: count of iterations

* warmup_batches: batches of warmup, default to 10000

* data_dir: path to OFRecord dataset

* data_part_num: number of files in the folder of OFRecord dataset

* use_fp16: use float16 or not

* use_boxing_v2: use boxing v2 or not

* loss_print_every_n_iter: print loss every n iterations

* model_save_every_n_iter: save the model every n iterations

* model_save_dir: path to save the model

* save_last_snapshot: whether save the model when training is finished

* model_load_dir: path to load the model

* log_dir LOG_DIR: specify the path of log

* seq_length: length of sequence, default to 512

* max_predictions_per_seq: default to 80

* num_hidden_layers: number of hidden layers, defaul to 24

* num_attention_heads: number of attentoion heads，default to 16

### Use Wikipedia + BookCorpus dataset
If it is necessary to carry out the pretraining of BERT from scratch, a large dataset should be used.

If necessary, we can download TFRecord dataset from [google-research BERT](https://github.com/google-research/bert) and then make OFRecord dataset from it by methods in the article [Loading and preparing OFRecord dataset](../extended_topics/how_to_make_ofdataset.md).

### OneFlow BERT model converted from Tensorflow Model
If you want to directly use the pretrained model for finetune tasks (such as the SQuAD shown below), you can consider downloading directly it from [google-research BERT](https://github.com/google-research/bert) and then use the script `convert_tf_ckpt_to_of.py` we provided to convert it to OneFlow model.

The conversion process is as follows:

Firstly, download and unzip a BERT pretrained model of specified version, eg: `uncased_L-12_H-768_A-12`.
```shell
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d uncased_L-12_H-768_A-12
```

And then, run commands below:
```shell
cd uncased_L-12_H-768_A-12/
cat > checkpoint <<ONEFLOW
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
ONEFLOW
```

It will create a file named `checkpoint` in the directory and write content below into it:
```
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
```

Now that the TensorFlow model directory to be converted is ready, the hierarchy is:
```shell
uncased_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── checkpoint
└── vocab.txt
```

And then we use `convert_tf_ckpt_to_of.py` to convert model to OneFlow format:
```bash
python convert_tf_ckpt_to_of.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12 \
  --of_dump_path ./uncased_L-12_H-768_A-12-oneflow
```
The above command saves the converted OneFlow format model in `./uncased_L-12_H-768_A-12-oneflow` directory for later use(eg: SQuAD).

## Finetune task: SQuAD
### Extend to SQuAD model
We only need to add a layer of `output` on the basis of BERT's backbone and modify the expression of loss. We can see the whole code in `squad.py`, and there are key modifications:
```python
def SQuADTrain():
    #... backbone = bert_util.BertBackbone()

    #add a fully-connected layer base on BERT
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

    #redefine the loss of SQuAD
        start_loss = _ComputeLoss(start_logits, start_positions_blob, seq_length)
        end_loss = _ComputeLoss(end_logits, end_positions_blob, seq_length)

        total_loss = 0.5*(start_loss + end_loss)

    return total_loss
```

We run the script below to start SQuAD training to get and save a initialized model.

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
There will be a initialized model in the path `./bert_regresssioin_test/of/last_snapshot`. We will merge it with pretrained BERT model and fintune it.

### Merge pretrained model into SQuAD
SQuAD is extended from pretrained model of BERT. We should merge the pretrained model into SQuAD according to the method introduced in this article[Loading and saving of model](../basics_topics/model_load_save.md).

```shell
cp -R ./bert_regresssioin_test/of/last_snapshot ./squadModel
cp -R --remove-destination ./dataset/uncased_L-12_H-768_A-12_oneflow/* ./squadModel/
```

### Problem on training times
There is a folder named `System-Train-TrainStep-xxx` in the path of pretrained model folder and the file named "out" contains the count if iterations. The `leraning rate` changes dynamically with the count of iterations.

In order to prevent training of finetuning from the saved iteration affecting, the binary data in the out file should be cleared to zero.
```shell
cd System-Train-TrainStep-xxx
xxd -r > out <<ONEFLOW
00000000: 0000 0000 0000 0000
ONEFLOW
```

If you are using a pretrained model transferred from TensorFlow, you can skip this step.

### Start SQuAD training
Start SQuAD training by running the script `run_suqad.py` with configuration below:

* use SQuAD model `./squadModel`

* use SQuAD v1.1 as training set

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

### Prediction and evaluatoin
In order to generate [Preidiction File](https://rajpurkar.github.io/SQuAD-explorer/), we should generate npy file fist. And then we use `write_predictions` function in [google BERT's run_squad.py](https://github.com/google-research/bert/blob/master/run_squad.py) to convert it to json format.

Run the script `run_squad_predict.py` to generate `all_results.npy`:
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
Attention: the `model_load_dir` should be the trained model of SQuAD.

After we get the `all_results.npy`file, run the script `npy2json.py` in the repository of [google bert](https://github.com/google-research/bert/)(the version of TensorFlow should be v1). The `npy2json.py` we provide is modified from google bert's `run_squad.py`:
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

Remember to set the `all_results_file` to the path of `all_results.npy` we obtained in the last step.

We will get `predictions.json` after that which can be evaluated by[evaluate-v1.1.py](https://rajpurkar.github.io/SQuAD-explorer/).

```bash
python evaluate-v1.1.py \
./dataset/dev-v1.1.json \
path/to/squad_base/predictions.json 
```

## Distributed training
As described when we introduce the command line options, we can start distributed training easily by adding the options `node_num` and `node_list`:

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
