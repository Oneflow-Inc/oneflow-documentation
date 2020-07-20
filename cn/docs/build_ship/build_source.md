

本文主要介绍：

* 如何同步oneflow及依赖的第三方库源码

* 如何编译oneflow

* 如何安装oneflow

* 如何将oneflow打包为whl包

* 支持XLA与TensorRT

## 编译源码

### 编译准备

编译oneflow需要使用`BLAS`库，在CentOS上，我们可以通过以下命令安装：

```shell
sudo yum -y install epel-release\
 && sudo yum -y install git gcc-c++ cmake3\
 openblas-devel kernel-devel-$(uname -r) nasm
```

如果你安装了`Intel MKL`库，确保环境变量已更新：

```shell
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
```

oneflow采用的cmake作为构建工具，cmake不正确，可能造成编译失败。推荐使用：
[https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz](https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz)

如果编译过程中出现问题，可以参考[常见编译问题汇总](troubleshooting.md)。

### 同步代码

通过以下命令，同步oneflow并更新第三方依赖源码。

```shell
git clone https://github.com/Oneflow-Inc/oneflow
git submodule update --init --recursive
```

以下命令可以同事同步oneflow与第三方代码，但速度可能较慢。
```shell
git clone https://github.com/Oneflow-Inc/oneflow --recursive
```


同步成功后，得到oneflow框架。

### 编译第三方库

先切换到编译路径build：

```shell
cd oneflow/build
```

之后的编译操作，如无特别说明，都在build目录下完成，不再重复说明。

在build目录下运行以下命令，编译第三方库：
```shell
cmake -DTHIRD_PARTY=ON .. 
make -j$(nproc)
```

### 编译oneflow

第三方库编译完成后，可以继续编译oneflow：

```shell
cmake .. \
-DTHIRD_PARTY=OFF \
-DPython_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
-DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

make -j$(nproc)
```

编译输出放置在`build/python_scripts/`路径下。

## 安装

只需要将包含有oneflow编译输出的路径，加入到PYTHONPATH环境变量即可完成安装：
```shell
export PYTHONPATH=your_path_to_source/oneflow/build/python_scripts:$PYTHONPATH
```

也可以使用在 **OneFlow仓库根目录下** 运行以下命令安装：
```shell
pip3 install -e . --user
```

## 生成whl包

在  **OneFlow仓库根目录下** 运行以下命令，可以将编译好的oneflow打包为whl文件。

```shell
python3 setup.py bdist_wheel
```

生成的`oneflow-x.x.x-cpxx-cpxxm-linux_x86_64.whl`文件，保存在仓库根目录下的`dist`文件夹中。

将whl拷贝到 **其它的** 机器上，可以省去编译链接过程，通过pip命令快速安装oneflow：
```shell
pip3 install oneflow-0.0.1-cp36-cp36m-linux_x86_64.whl
```


## 支持XLA与TensorRT

oneflow也支持`TensorFlow XLA`或者`TensorRT`后端加速引擎。技术细节可参考这篇[文档](https://github.com/Oneflow-Inc/oneflow/blob/develop/oneflow/xrt/README.md)。

本节介绍如何在编译中加入XLA或者TensorRT支持。

### 编译时加入XLA支持

首先，安装Bazel。
从[这里](https://docs.bazel.build/versions/1.0.0/bazel-overview.html) 下载并安装bazel。我们推荐使用0.24.1版本。可以通过以下命令查看bazel版本号。

```shell
bazel version
```

为了支持XLA，我们需要在编译第三方库时：
```shell
cmake -DWITH_XLA=ON -DTHIRD_PARTY=ON ..
make -j$(nproc)
```

该过程会下载XLA需要的依赖并编译安装。如果下载出错，则需要 **重新安装cmake** 并清空CMakeCache.txt，再重新编译第三方库。

第三方库编译完成后，采用以下命令编译`OneFlow`：

```shell
cmake .. \
-DWITH_XLA=ON \
-DTHIRD_PARTY=OFF \
-DPython_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
-DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

make -j$(nproc)
```

### 编译时加入TensorRT的支持

首先，下载TenSortRT(>=6.0)并且解压。
之后，进入到oneflow源码的`build`目录下，编译第三方库，加上TensorRT的支持：

```shell
cmake -DWITH_TENSORRT=ON -DTENSORRT_ROOT=your_tensorrt_path -DTHIRD_PARTY=ON ..
make -j$(nproc)
```

最后，通过以下命令编译oneflow：

```shell
cmake .. \
-DWITH_TENSORRT=ON \
-DTENSORRT_ROOT=your_tensorrt_path \
-DTHIRD_PARTY=OFF \
-DPython_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
-DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

make -j$(nproc)
```
