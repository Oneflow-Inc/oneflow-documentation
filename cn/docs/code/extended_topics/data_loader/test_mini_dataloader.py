import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import numpy as np

flow.config.load_library("final_output.so")

def MiniDecoder(
    input_blob,
    name = None,
):
    if name is None:
        name = "Mini_Decoder_uniqueID"
    return (
        flow.user_op_builder(name)
        .Op("mini_raw_decoder")
        .Input("in", [input_blob])
        .Output("x")
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

def MiniReader(
    ofrecord_dir: str,
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    random_shuffle: bool = False,
    shuffle_buffer_size: int = 1024,
    shuffle_after_epoch: bool = False,
    name = None,
):
    if name is None:
        name = "Mini_Reader_uniqueID"

    return (
        flow.user_op_builder(name)
        .Op("MiniReader")
        .Output("out")
        .Attr("data_dir", ofrecord_dir)
        .Attr("data_part_num", data_part_num)
        .Attr("batch_size", batch_size)
        .Attr("part_name_prefix", part_name_prefix)
        .Attr("random_shuffle", random_shuffle)
        .Attr("shuffle_buffer_size", shuffle_buffer_size)
        .Attr("shuffle_after_epoch", shuffle_after_epoch)
        .Attr("part_name_suffix_length", part_name_suffix_length)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

config = flow.function_config()
config.default_data_type(flow.double)

@flow.global_function("train", config)
def test_job() -> tp.Numpy:
    batch_size = 10
    with flow.scope.placement("cpu", "0:0"):
        miniRecord = MiniReader(
            "./",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=3,
            random_shuffle=True,
            shuffle_after_epoch=True,
        )

        x, y = MiniDecoder(
             miniRecord, name="d1"
         )

        initializer1 = flow.random_uniform_initializer(-1/28.0, 1/28.0)
        hidden = flow.layers.dense(
                    x,
                    500,
                    activation=flow.nn.relu,
                    kernel_initializer=initializer1,
                    bias_initializer=initializer1,
                    name="dense1",
                )
        initializer2 = flow.random_uniform_initializer(-np.sqrt(1/500.0), np.sqrt(1/500.0))
        logits = flow.layers.dense(
        hidden, 1, kernel_initializer=initializer2, bias_initializer=initializer2, name="dense2"
        )
        loss = (y - logits)*(y-logits)

        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
        flow.optimizer.Adam(lr_scheduler).minimize(loss)
        return loss


if __name__ == "__main__":
    ckpt = flow.train.CheckPoint()
    ckpt.init()
    loss = test_job()
    for i in range(0, 5000):
        loss = test_job()
        if i % 100 == 0 : print("{}/{}:{}".format(i, 5000, loss.mean()))
    ckpt.save("./test_model")
