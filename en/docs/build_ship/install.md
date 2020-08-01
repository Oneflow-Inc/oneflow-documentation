## Install OneFlow launch edition

Use following commands to install the latest version of OneFlow which support CUDA:

python3 -m pip install oneflow_cu102 --user
```

你也可以安装支持其他 CUDA 版本的 OneFlow ，在下面选择其一即可

```python
python3 -m pip install oneflow_cu101
python3 -m pip install oneflow_cu100
python3 -m pip install oneflow_cu92
python3 -m pip install oneflow_cu91
python3 -m pip install oneflow_cu90
```

We recommend use the latest version of CUDA. Please update your Nvidia driver to 440.33 or higher and proceed installation.

`oneflow_cu102`

## Common issue:

* If installing failed, please try upgrade `pip` ：
```shell
python3 -m pip install --upgrade pip
python3 -m pip install oneflow_cu102
```

* If access is denied( `site-packages is not writeable` ), it is highly possible of do not have administrator authentication. Could add `--user` (same in installing OneFlow)：
```shell
python3 -m pip install --upgrade pip --user
python3 -m pip install oneflow_cu102 --user
```

* If it is too slow when downloading, you can try following command:
```shell
python3 -m pip install oneflow_cu102 -i https://pypi.douban.com/simple
```

## Compile from source and install OneFlow

If you want to install OneFlow by compile the source code. Please reference to OneFlow source code repository [README](https://github.com/Oneflow-Inc/oneflow/blob/develop/README.md)，Before compile OneFlow source code, highly recommended to read [Troubleshooting](https://github.com/Oneflow-Inc/oneflow/blob/develop/docs/source/troubleshooting.md).

## Install OneFlow Nightly version

We also provide daily updates OneFlow Nightly version. Suit for user who want you try new functions of OneFlow. Not recommended in production environments. Use following command to install: 

```shell
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu102 --user
```

## QQ channel
If you meet any issue in installation, you are welcome to share you issue in QQ channel and let OneFlow fans to discuss：

**QQ channel 331883 or scan QR code **

![qq group](../contribute/imgs/qq_group.png)
