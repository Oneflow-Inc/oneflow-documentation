## 安装 OneFlow 稳定发布版

使用以下命令安装 OneFlow 最新的支持CUDA的稳定版本：

```shell
python3 -m pip install -f https://release.oneflow.info oneflow==0.4.0+cu102
```

使用以下命令安装 OneFlow 最新 master 分支（不建议生产环境下使用）：
```shell
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu102
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

Stable:
```
python3 -m pip install --find-links https://release.oneflow.info oneflow==0.4.0+[PLATFORM] --user
```

Nightly:
```
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
```

其中 `[PLATFORM]` 可以是:

| Platform |CUDA Driver Version| Supported GPUs |
|---|---|---|
| cu112  | >= 450.80.02  | GTX 10xx, RTX 20xx, A100, RTX 30xx |
| cu111  | >= 450.80.02  | GTX 10xx, RTX 20xx, A100, RTX 30xx |
| cu110, cu110_xla  | >= 450.36.06  | GTX 10xx, RTX 20xx, A100|
| cu102, cu102_xla  | >= 440.33  | GTX 10xx, RTX 20xx |
| cu101, cu101_xla  | >= 418.39  | GTX 10xx, RTX 20xx |
| cu100, cu100_xla  | >= 410.48  | GTX 10xx, RTX 20xx |
| cpu  | N/A | N/A |

## 交流QQ群
安装或使用过程遇到问题，欢迎入群与众多 OneFlow 爱好者共同讨论交流：

**加QQ群 331883 或扫描二维码**

![qq group](../contribute/imgs/qq_group.png)
