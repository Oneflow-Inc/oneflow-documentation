# feed_numpy.py
import numpy as np
import oneflow as flow
import oneflow.typing as oft

@flow.global_function(flow.function_config())
def test_job(images:oft.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((32,), dtype=flow.int32)):
    # do something with images or labels
    return images, labels


if __name__ == '__main__':
    images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
    labels_in = np.random.randint(-10, 10, (32,)).astype(np.int32)
    images, labels = test_job(images_in, labels_in).get()
    print(images.shape, labels.shape)
