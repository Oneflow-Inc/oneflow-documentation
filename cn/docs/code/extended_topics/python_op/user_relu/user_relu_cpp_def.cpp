#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

REGISTER_USER_OP("user_relu_forward")
  .Input("x")
  .Output("y")
  .SetTensorDescInferFn(
      [](user_op::InferContext *ctx) -> Maybe<void> {
        *ctx->Shape4ArgNameAndIndex("y", 0) =
            *ctx->Shape4ArgNameAndIndex("x", 0);
        *ctx->Dtype4ArgNameAndIndex("y", 0) =
            *ctx->Dtype4ArgNameAndIndex("x", 0);
        return Maybe<void>::Ok();
      });

REGISTER_USER_OP("user_relu_backward")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr<std::string>("device_sub_tag", "py")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("user_relu_forward")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      const auto& grad_op_func = [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("user_relu_backward")
            .InputBind("y", ctx->FwOp().output("y", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      };
      ctx->DefineOp(grad_op_name, grad_op_func);

      const auto& dx_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dx", 0);
      };
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), dx_get_func);
    });

}  // namespace
}  // namespace oneflow
