#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template <typename T>
void MyOp(DeviceCtx *ctx, const int64_t n, const T *x, T *y) {
  for (int64_t i = 0; i != n; ++i) {
    y[i] = 3*x[i]*x[i];
  }
}

template <DeviceType device_type, typename T>
class MyOpKernel final : public user_op::OpKernel {
public:
  MyOpKernel() = default;
  ~MyOpKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    MyOp<T>(ctx->device_ctx(),
           in_tensor->shape().elem_cnt(),
           in_tensor->dptr<T>(), 
           out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MYOP_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("myop")                     \
      .SetCreateFn<MyOpKernel<device, dtype>>()      \
      .SetIsMatchedHob(                              \
          (user_op::HobDeviceTag() == device) &     \
          (user_op::HobDataType("out", 0)            \
            == GetDataType<dtype>::value));

REGISTER_MYOP_KERNEL(DeviceType::kCPU, float)
REGISTER_MYOP_KERNEL(DeviceType::kCPU, double)
} // namespace

} // namespace oneflow
