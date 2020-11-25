# Oneflow-Documentation

## Documentation

此项目为 OneFlow 官方在线文档的 [github 仓库](https://github.com/Oneflow-Inc/oneflow-documentation)，你也可直接浏览我们的在线文档：http://docs.oneflow.org/

在线文档内容全面，不仅有 [OneFlow 系统设计](http://docs.oneflow.org/basics_topics/essentials_of_oneflow.html)的介绍，还包括以下模块：

- [【 首页】](http://docs.oneflow.org/index.html)
- [【 快速上手】](http://docs.oneflow.org/quick_start/install.html)
- [【基础专题】](http://docs.oneflow.org/basics_topics/data_input.html)
- [【拓展专题】](http://docs.oneflow.org/extended_topics/job_function_define_call.html)
- [【高级应用实例】](http://docs.oneflow.org/adv_examples/resnet.html)
- [【API】](https://oneflow.readthedocs.io/en/master/)
- [【OneFlow 开源计划】](http://docs.oneflow.org/contribute/intro.html)

本仓库包含中文文档和英文文档，分别存放于cn、en文件夹下，文档目录结构树：

```shell
.
├── quick_start             快速上手
├── basics_topics           基础专题
├── extended_topics         拓展专题
├── code                    文档示例代码
│   ├── basics_topics
│   ├── extended_topics
│   └── quick_start
├── adv_examples            高级应用实例
├── extended_dev            待发布文章
└── contribute              OneFlow开源计划
```



## API

我们的 API 在线文档部署在 ReadTheDocs 上：

https://oneflow.readthedocs.io/en/master/



## Benchmark

我们还在 Benchmark 项目中提供了一系列的基准模型，包括：

- NLP 相关的 **BERT**
- CTR 相关的 **WideDeepLearning**
- CV 相关的 CNN 分类模型： **Resnet50** 、 **VGG-16** 、 **Inception-v3** 、 **Alexnet**

github地址：https://github.com/Oneflow-Inc/OneFlow-Benchmark


## DLPerf
此外，我们还开源了DLPerf仓库：https://github.com/Oneflow-Inc/DLPerf

该仓库对各个框架的多个经典模型如ResNet50、BERT等做了训练性能评测，给出了各大框架的训练速度、加速比等指标。DLPerf中的评测覆盖了单机单卡/多机多卡、FP32/混合精度、XLA等情况，具体数据和报告可参考：[dlperf_benchmark_test_report_v1_cn](https://github.com/Oneflow-Inc/DLPerf/blob/master/reports/dlperf_benchmark_test_report_v1_cn.md)