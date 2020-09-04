/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("mini_decoder")
    .Input("in")
    .Output("x")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* out_tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);

      *in_tensor->mut_data_type() = DataType::kDouble;

      *out_tensor_x->mut_shape() = Shape({in_tensor->shape().At(0), 1});
      *out_tensor_x->mut_data_type() = DataType::kDouble;

      *out_tensor_y->mut_shape() = Shape({in_tensor->shape().At(0), 1});
      *out_tensor_y->mut_data_type() = DataType::kDouble;
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL(in_modifier);
      in_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
