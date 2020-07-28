# DCGAN tutorial



## Introduction

Generative Adversarial Networks (GANs) are a type of generative network that learns specific data distribution through a zero-sum game of two networks.DCGAN is a Generative Adversarial Network based on convolution/deconvolution operations, which is widely used in the field of image generation.

This example will mainly demonstrate how to run the DCGAN network in Oneflow, without focusing on the principles and details of generating a confrontation network.If you are interested, you can refer to:

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

- [NLPS 2016 tutorial:generative adversarial networks](https://arxiv.org/abs/1511.06434)



## The core code of this example is in the dcgan. py file. The model structure and parameters refer to the official example of tensorflow

The core code of this example is in the `dcgan.py` file, and the model structure and parameters refer to tensorflow's [official example](https://www.tensorflow.org/tutorials/generative/dcgan)

Through the following code, you can run a simple alignment test to ensure that the results of the oneflow model are consistent with the results of tensorflow

```python
dcgan = DCGAN()
dcgan.compare_with_tensorflow()
```



## Dataset Preparation

The example provides a data set download script, run `download.py` to download the mnist data set, and the dataset is saved in the directory `./data/minst` by default

```bash
python download.py mnist
```



## Training

After the data set is prepared, you can start DCGAN training through the `train` method of the DCGAN instance

```python
dcgan.train(epochs=2)
```

It will output the generated images every `self.eval_interval` batch during training.

![1](imgs/1.png)

## Save to gif

After completing the training, the image can be exported as a moving image through the save_to_gif method of the DCGAN instance

```python
dcgan.save_to_gif()
```