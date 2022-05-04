# Run Python Code of Markdown Files

本目录下的程序，可以读取 YAML 文件中的配置，提取并运行 markdown 文件中的 Python 代码。

- [配置 YAML 文件](#配置-YAML-文件)
- [读取 YAML 中的设置并运行](#读取-YAML-中的设置并运行)
- [读取所有 YAML 中的设置并运行](#读取所有-YAML-中的设置并运行)
- [测试时修改 markdown 中的代码](#测试时修改-markdown-中的代码)
- [run_by_yamls.py 选项说明](#run_by_yamls.py-选项说明)

## 配置 YAML 文件

配置文件（YAML 格式）放置在 [configs](./configs/) 目录下。

以 [basics_02_tensor.yml](./configs/basics_02_tensor.yml) 为例，查看其中的设置项：

```yaml
- file_path: cn/docs/basics/02_tensor.md
  run:
    - all
```

`file_path` 指定 markdown 文件（相对于本仓库）的路径。

`run` 指定需要运行的 Python 代码块。所谓 “Python 代码块”，指的是 markdown 中以 \`\`\`python 开头的代码块。“all” 表示依次运行其中所有的 Python 代码块。

`run` 的值也可以是一个代码块序号组成的 list，如：

```yaml
- file_path: cn/docs/basics/02_tensor.md
  run:
    - [0, 1, 2]
```

表示只运行 `02_tensor.md` 文件中的前 3 个 Python 代码块。

通过 `run_by_yamls.py` 的 `--markdown` 选项，可以查看一个 markdown 文件中的所有 Python 代码块及其序号。

```shell
python3 run_by_yamls.py --markdown cn/docs/basics/04_build_network.md
```

以上命令输出：

```text
=============CODE 0=============
import oneflow as flow
import oneflow.nn as nn


=============CODE 1=============
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

...

=============CODE 5=============
class MySeqModel(nn.Module):
    def __init__(self):
        super(MySeqModel, self).__init__()
        self.seq = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)
```

一个 markdown 文件中，在运行时可以运行多组（次）测试。如：

```yaml
- file_path: cn/docs/cookies/transfer_learning.md
  run: 
    - [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    - [0, 1, 2, 3, 4, 5, 6, 7, 9, 11]
    - [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
```

以上配置，针对 `transfer_learning.md` 文件，会运行 3 组 Python 代码。

一个 YAML 文件中，可以针对任意多个 markdown 文件进行配置，如：

```yaml
- file_path: cn/docs/basics/02_tensor.md
  run:
    - all

- file_path: en/docs/basics/02_tensor.md
  run:
    - all
```


## 读取 YAML 中的设置并运行

`run_by_yamls.py` 的 `--yaml` 选项，可以读取指定的 YAML，提取对应 markdown 文件中的代码，并运行。如：

```shell
python3 run_by_yamls.py --yaml ./configs/basics_02_tensor.yml
```

## 读取所有 YAML 中的设置并运行

如果 `run_by_yamls.py` 运行时不带任何选项，则表示读取本仓库 `scripts/markdown_ci/configs/` 目录下的所有 YAML 文件，并运行对应 markdown 文件中的代码。

```bash
python3 run_by_yamls.py
```


## 测试前修正 markdown 中的代码

每一个 markdown 文件，还可以配置一个 `hook` 项，如：

```yaml
- file_path: cn/docs/basics/01_quickstart.md
  run:
    - all
  hook: | # hook(index, codetext)
      if index == 8:
        code = code.replace("epochs = 5", "epochs = 1")
      return code
```

`hook` 中的值为回调函数的实现代码，该函数会接受 `index` 和 `code` 两个参数，分别指代码块的序号和代码块的内容，该函数的返回值，将被当作最终运行的代码块内容。
如以上的配置中，将 8 号代码块中的 `epochs = 5` 改成 `epochs = 1`，用以减少运行时间。

## run_by_yamls.py 选项说明

```shell
python run_by_yamls.py -h
```

```text
usage: run_by_yamls.py [-h] [--markdown MARKDOWN] [--output OUTPUT]
                       [--yaml YAML] [--configs CONFIGS]

read config yaml files and run realted code

optional arguments:
  -h, --help           show this help message and exit
  --markdown MARKDOWN  the input markdown file
  --output OUTPUT      if not None, output will be written to the path
  --yaml YAML          the path of yaml file. eg: ./sample.yaml
  --configs CONFIGS    config dir where yaml files exists, markdown_ci/configs
                       by default.
```

- `--output` 配合 `--markdown` 使用，如果指定了 `--output`，则提取的 Python 代码块内容，会重定向到文件。
- `--config` 用于指定存放 YAML 文件的路径，方便测试。

## 如何查看错误信息

当运行报错时，会打印出错误信息：

```text
====RUN CODE IN MARKDOWN====: python3 run_markdown_codes.py --markdown_file /workspace/oneflow-documentation/cn/docs/basics/04_build_network.md --index all
...
    ****EXEC ERROR****
    markdown file: /workspace/oneflow-documentation/cn/docs/basics/04_build_network.md
    codeblock index: 2
    Code:b'X = flow.ones(1, 28, 28)\nlogits = net(X)\npred_probab = nn.Softmax(dim=1)(logits)\ny_pred = pred_probab.argmax(1)\nprint(f"Predicted class: {y_pred}")\n'

Traceback (most recent call last):
  File "run_markdown_codes.py", line 21, in run_block_item
    exec(code, globals(), globals())
  File "<string>", line 1, in <module>
NameError: name 'flow' is not defined

During handling of the above exception, another exception occurred:

...
```

其中 `====RUN CODE IN MARKDOWN====` 告知了正在提取并运行哪个 markdown 文件。

`****EXEC ERROR****` 告知出错代码块的序号（`codeblock index: 2`），代码块的内容 `Code: ...`。
