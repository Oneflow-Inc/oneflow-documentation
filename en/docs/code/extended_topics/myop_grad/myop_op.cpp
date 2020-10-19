#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("myop").Input("in").Output("out").SetTensorDescInferFn(
    [](user_op::InferContext *ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) =
          *ctx->Shape4ArgNameAndIndex("in", 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) =
          *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("myop").SetBackwardOpConfGenFn(
    [](user_op::BackwardOpConfContext* ctx) {

      const auto op1_name = ctx->FwOp().op_name() + "_grad1";
      
      // 算子 op1_name 用于计算 myop.in*(myop.out的梯度)
      ctx->DefineOp(op1_name, 
        [&ctx](user_op::BackwardOpBuilder& builder) {
          return builder.OpTypeName("multiply")
              .InputBind("x", ctx->FwOp().input("in", 0)) //multiply.x <- myop.in
              .InputBind("y", ctx->FwOp().output_grad("out", 0)) //multiply.y <- myop.out的梯度
              .Output("out")
              .Build();
        });

      const auto op2_name = ctx->FwOp().op_name() + "_grad2";
      // 算子 op2_name 用于计算 6*op1_name
      ctx->DefineOp(op2_name, 
        [&ctx, &op1_name](user_op::BackwardOpBuilder& builder) {
          return builder.OpTypeName("scalar_mul")
              .InputBind("in", ctx->GetOp(op1_name).output("out", 0))
              .Attr("has_float_operand", true)
              .Attr("has_int_operand", false)
              .Attr("float_operand", static_cast<double>(6))
              .Attr("int_operand", static_cast<int64_t>(6))
              .Output("out")
              .Build();
        });
      
      // (myop.in的梯度) <- op1_name.out
      ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), 
        [&ctx, &op2_name]() -> const std::string& {
          return ctx->GetOp(op2_name)
                .output("out", 0);
        });
  });
} // namespace

} // namespace oneflow
