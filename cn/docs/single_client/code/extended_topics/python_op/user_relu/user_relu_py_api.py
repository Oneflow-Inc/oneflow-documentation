import oneflow as flow

def user_relu_forward(x):
    op = (
        flow.user_op_builder("myrelu")
        .Op("user_relu_forward")
        .Input("x", [x])
        .Output("y")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
