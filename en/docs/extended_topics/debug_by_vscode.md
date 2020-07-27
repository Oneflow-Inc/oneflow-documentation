The developing environment of OneFlow is Linux. If we want to use GUI developing or configure OneFlow. We can use VS code withe a Remote - SSH connection to server.

If you are not familiar with VScode please reference to [official documentation](https://code.visualstudio.com/docs).

This article:

* How to edit the  `Debug` version of OneFlow.

* The necessary extension packages of VS code and installed guidelines.

### Edit the Debug version of OneFlow.

If use the  `Release` version of OneFlow. Because of editor optimization issue, the issue that don't correspond to the actual position and source line may occur in the process of debugging programs.

Thus, we need to edit `Debug` version of OneFlow. And need to generate json files need by clangd.

When running cmake, we need add flag of  `Debug` and `CMAKE_EXPORT_COMPILE_COMMANDS`

```shell
cmake .. \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1
```
Above code:

* `-DCMAKE_BUILD_TYPE=Debug`  choose the version of Debug.

* `-DCMAKE_EXPORT_COMPILE_COMMANDS`  will generate the  `compile_commands.json` files need by clangd in  `build`.

### Remote - SSH
Use the  Remote SSH of  VS Code can use SSH connects to a server.

![RemoteSSH](imgs/plugin-remote-ssh.png)

We can operate the OneFlow on server and use Remote SSH connect  VS Code. Thus **it can let user operated just like local environment**.

After finish Remote - SSH installation, press F1 then enter `Remote-SSH: Connect to Host..` to search. After config ssh then you can connect to server.

After use Remote - SSH, in plug column, will automatically classify the local and remote. If detected the need some plug need install on remote server. It will show in grey and with a button called **Install in SSH: remote server name**. Click on that can install the corresponding plugs.

![remotePlugin](imgs/plugin-remote-ssh-install.png)

Like the figure show, we have already installed Python, clangd and NativeDebug. In order to support remote configuration.

But the remote console didn’t installed Go and HTML CSS Suport plug.


### clangd
After sample configuration, clangd can support convenient of code completion and symbols jump.

Before configure clangd, we need make sure:

* Already use editor generated `compile_commands.json` file.

* Already use Remote - SSH install clangd on remote console.

* 建议 **不要** 安装 VS Code 推荐的 ms-vscode.cpptools C/C++ 插件，因为 clangd 有可能与之冲突

#### 安装 clangd 程序
VS Code 上的插件，是通过与clangd服务程序交互，获取解析信息并显示的。VS Code 上的插件，是通过与clangd服务程序交互，获取解析信息并显示的。因此除了安装 VS Code 上的 clangd 插件外，我们还需要在 **OneFlow源码所在的主机上** （本文中为远程Linux主机）安装clangd服务程序。

我们将采用 **下载 zip 文件并解压** 的方式安装，更多安装方法，可以参考 [clangd 官方文档](https://clangd.llvm.org/installation.html)。

首先，在[这个页面](https://github.com/clangd/clangd/releases/)下载与我们系统平台对应的clangd压缩包，并解压。 解压后可先运行 clangd 测试，确保能正常运行后再进行后续配置。 解压后可先运行 clangd 测试，确保能正常运行后再进行后续配置。

```shell
/path/to/clangd/bin/clangd --help
```

#### 配置 VS Code 中的 clangd 插件

将 build 目录下的 `compile_commands.json` 文件软链接到 OneFlow 的源码根目录下，在OneFlow的源码根目录下：

```shell
ln -s ./build/compile_commands.json compile_commands.json
```

然后`Ctrl+Shift+P` (macOS 下 `command+shift+p`)，找到 `Open Remote Settings` 选项，打开 `settings.json` 配置文件，在其中加入以下配置：

```json
    "clangd.path": "/path/to/bin/clangd",
    "clangd.arguments": [
        "-j",
        "12",
        "-clang-tidy"
    ]
```
`clangd.arguments`的意义及更多参数选项，可查阅`clangd --help`。

#### 使用 clangd
在 VS Code 的 View->Output 面板，下拉菜单中选择 "Clang Language Server"，可以看到 clangd 的解析输出，解析完成后。选择 C/C++ 源码中的符号，可以实现跳转。选择 C/C++ 源码中的符号，可以实现跳转。

`Ctrl+Shift+P` (macOS 下 `command+shift+P`) 中通过`@符号名`或`#符号名`可以分别实现当前文件内查找符号，或工程范围内查找符号。



### native debug
`Ctrl + Shift + D` (macOS 下 `command+shift+D`) 或者点击 activity bar 的 Run 按钮，进入到 Run 视图。

![Run View](imgs/run-view.png)

选择 `Create a launch.json file`，选择 gdb 模板。 ![gdb](imgs/gdb-select.png)

然后设置相关参数：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "lenet", //自定义任务名
            "type": "gdb",
            "request": "launch",
            "target": "/home/yaochi/.conda/envs/ycof/bin/python3", //python路径
            "arguments": "lenet_train.py", //脚本
            "cwd": "/home/yaochi/of_example", //脚本所在路径
            "valuesFormatting": "parseText"
        }
    ]
}
```

设置断点后，F5 启动调试： ![调试截图](imgs/debug_snapshot.png)

### 其它

* 如果 VS Code 下载插件速度过慢，可以按照[官方文档](https://code.visualstudio.com/docs/setup/network)的步骤切换 `hostname` 或者设置代理。

* 关于 clangd 安装配置的[官方介绍](https://clang.llvm.org/extra/clangd/Installation.html)

* 关于 VS Code 的调试设置的[官方介绍](https://code.visualstudio.com/docs/editor/debugging)

* clangd 的最新版本可能对 glibc 版本要求过高，导致报缺少库的错误。

```shell
./bin/clangd: /lib64/libc.so.6: version `GLIBC_2.18' not found (required by ./bin/clangd)
```

此时可以下载其它更低 clangd 的版本，早期版本的 clangd 需要到 [LLVM官网](https://releases.llvm.org/download.html) 下载整个LLVM工具链，其中包含有 clangd。
