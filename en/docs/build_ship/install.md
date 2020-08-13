## Installation

To install latest release of OneFlow with CUDA support:

```
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu102 --user
```
    
## System requirements：

  - Python >= 3.5
  - CUDA Toolkit Linux x86_64 Driver
  
    | OneFlow |CUDA Driver Version|
    |---|---|
    | oneflow_cu102  | >= 440.33  |
    | oneflow_cu101  | >= 418.39  |
    | oneflow_cu100  | >= 410.48  |
    | oneflow_cu92  | >= 396.26  |
    | oneflow_cu91  | >= 390.46  |
    | oneflow_cu90  | >= 384.81  |

If there are errors like "cannot find", please try to update `pip`:
```shell
python3 -m pip install --upgrade --user pip
```
## Build from source

If you want to install OneFlow by building from source. Please refer to [README](https://github.com/Oneflow-Inc/oneflow/blob/develop/README.md) of OneFlow. Please read the [Troubleshooting](https://github.com/Oneflow-Inc/oneflow/blob/develop/docs/source/troubleshooting.md) for common issues you might encounter when compiling and running OneFlow.

## Install OneFlow with legacy CUDA support

To install OneFlow with legacy CUDA support, run one of:
```
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu101 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu100 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu92 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu91 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu90 --user
```

## QQ channel 
If you encounter any problems during installation, you are welcome to discuss in QQ channel or [submit issues to OneFlow](https://github.com/Oneflow-Inc/oneflow/issues).

**QQ channel ID: 331883 or scan QR code below**

![qq group](../contribute/imgs/qq_group.png)
