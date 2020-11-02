## 安装 OneFlow 稳定发布版

使用以下命令安装 OneFlow 最新的支持CUDA的稳定版本：

```shell
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu102 --user
```

使用以下命令安装 OneFlow 最新 CPU-ONLY 的版本：
```shell
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cpu --user
```

如果提示 **找不到** 对应版本，请尝试升级 `pip`：
```shell
python3 -m pip install --upgrade --user pip
```

国内用户可以使用国内镜像加速
```
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
详细说明可见 [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)。


系统要求：

* Python >= 3.5

* CUDA 驱动要求详情见 OneFlow 源码仓库 [README](https://github.com/Oneflow-Inc/oneflow/#system-requirements)

## 从源码编译安装 OneFlow

如果你希望通过编译源码安装 OneFlow，可以参考 OneFlow 源码仓库的 [README](https://github.com/Oneflow-Inc/oneflow/blob/develop/README.md)，在编译 OneFlow 源码之前，强烈推荐先阅读 [Troubleshooting](https://github.com/Oneflow-Inc/oneflow/blob/develop/docs/source/troubleshooting.md)。

## 安装 OneFlow with legacy CUDA
支持其它较早版本 CUDA 的 OneFlow 的安装方法如下：
```shell
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu101 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu100 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu92 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu91 --user
python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu90 --user
```

## 交流QQ群
安装或使用过程遇到问题，欢迎入群与众多 OneFlow 爱好者共同讨论交流：

**加QQ群 331883 或扫描二维码**

![qq group](../contribute/imgs/qq_group.png)
