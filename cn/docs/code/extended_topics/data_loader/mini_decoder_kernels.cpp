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
#include "oneflow/core/thread/thread_manager.h"

#include <cstdint>


namespace oneflow {

class MiniDecoderKernel final : public user_op::OpKernel {
 public:
  MiniDecoderKernel() = default;
  ~MiniDecoderKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_blob_y = ctx->Tensor4ArgNameAndIndex("y", 0);

    int64_t record_num = in_blob->shape().At(0);

    const double* input = in_blob->dptr<double>();
    double* out_dptr_x = out_blob_x->mut_dptr<double>();
    double* out_dptr_y = out_blob_y->mut_dptr<double>();

    MultiThreadLoop(record_num, [&](size_t i){
      *(out_dptr_x + i) = *(input+i*2);
      *(out_dptr_y + i) = *(input+i*2 + 1);
    });

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("mini_decoder")                                              \
    .SetCreateFn<MiniDecoderKernel>()                                             \
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                           \
                      & (user_op::HobDataType("in", 0) == DataType::kDouble)      \
                      & (user_op::HobDataType("x", 0) == DataType::kDouble)       \
                      & (user_op::HobDataType("y", 0) == DataType::kDouble));
}  // namespace oneflow
