# 模型训练可视化
Oneflow支持将训练生成的中间结果以日志文件的形式保存到本地，可视化后端通过实时读取日志文件，将训练过程产生的数据实时展示到可视化前端。

目前，Oneflow支持的可视化类型分为以下几种：

| 可视化类型 | 描述                     |
| ---------- | ------------------------ |
| 模型结构   | 结构图、计算图(后续支持) |
| 标量数据   | 标量数据                 |
| 媒体数据   | 文本、图像               |
| 统计分析   | 数据直方图、数据分布图   |
| 降维分析   | 数据降维                 |
| 超参分析   | 超参数                   |
| 异常检测   | 异常数据检测             |

本文将介绍
* 如何初始化summary日志文件

* 如何生成结构图日志

* 如何生成标量数据日志

* 如何生成媒体数据日志

* 如何生成统计分析数据日志

* 如何生成降维分析日志

* 如何生成超参分析日志

* 如何生成异常检测日志

本文中提到的可视化日志具体使用方式可参考test_summary.py 文件，

具体可视化效果参考[之江天枢人工智能开源平台](http://tianshu.org.cn/?/course)用户手册可视化部分。

## 初始化

首先定义一个用于存放日志文件的目录 logdir, 我们定义以下计算函数调用 flow.summary.create_summary_writer接口创建日志文件写入方法。
```python
# CreateWriter.py
    @flow.global_function(function_config=func_config)
    def CreateWriter():
        flow.summary.create_summary_writer(logdir)
```
再调用CreateWriter()这个函数，就可以完成summary writer对象的创建啦
