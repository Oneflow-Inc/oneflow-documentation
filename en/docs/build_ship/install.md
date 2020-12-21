## Install OneFlow Stable Version 

Install the latest stable version of OneFlow with CUDA support using the following command:

```shell
python3 -m pip install --find-links https://release.oneflow.info oneflow_cu102 --user
```

Install the latest CPU-ONLY version of OneFlow using the following command:
```shell
python3 -m pip install --find-links https://release.oneflow.info oneflow_cpu --user
```

Install the latest version of the OneFlow master branch using the following command (not recommended for use in production environments):
```shell
python3 -m pip install --find-links https://staging.oneflow.info/branch/master oneflow_cu102 --user
```

If you are informed that the corresponding version cannot be found, please try upgrading `pip`:
```shell
python3 -m pip install --upgrade --user pip
```

Chinese users can use the domestic mirror to accelerate:
```
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
Detailed instructions can be found in the [pypi mirror help](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)。


System Requirements:

* Python >= 3.5

* CUDA driver requirements are available in the OneFlow source code repository [README](https://github.com/Oneflow-Inc/oneflow/#system-requirements)

## Build from source

If you want to install OneFlow by building from source, please refer to [README](https://github.com/Oneflow-Inc/oneflow/blob/develop/README.md). Also refer to [Troubleshooting](https://github.com/Oneflow-Inc/oneflow/blob/develop/docs/source/troubleshooting.md) for common issues you might encounter when compiling and running OneFlow.

## Install OneFlow with legacy CUDA support

To install OneFlow with legacy CUDA support, run one of the following command:
```
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu101 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu100 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu92 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu91 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu90 --user
```

## QQ channel 
If you encounter any problems during the installation and want for help, please join the QQ channel or [submit issues on Github](https://github.com/Oneflow-Inc/oneflow/issues).

**QQ channel ID: 331883 or scan QR code below**

![qq group](../contribute/imgs/qq_group.png)
