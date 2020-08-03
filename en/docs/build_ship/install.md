## Install OneFlow launch edition

Use following commands to install the latest version of OneFlow:
```shell
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu102 --user
```
System requirements：

* Python >= 3.5

* Nvidia Linux x86_64 driver version >= 440.33

If error shows that **cannot find** corresponding version, please try update `pip`：
```shell
python3 -m pip install --upgrade --user pip
```
## Install by compile source code of OneFlow

If you want to install by compile source code. Please reference to source code repository of OneFlow [README](https://github.com/Oneflow-Inc/oneflow/blob/develop/README.md). Before install by compile source code. We recommend you to read [Troubleshooting](https://github.com/Oneflow-Inc/oneflow/blob/develop/docs/source/troubleshooting.md).



## Install OneFlow with legacy CUDA
Support earlier version of CUDA install commands：
```shell
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu101 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu100 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu92 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu91 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu90 --user
```

## QQ channel 
If you meet any issue during installation, you are welcome to share you issue in QQ channel and let OneFlow fans to discuss：

**QQ channel ID: 331883 or scan QR code below**

![qq group](../contribute/imgs/qq_group.png)
