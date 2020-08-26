#include "oneflow/core/framework/framework.h"
#include <cub/cub.cuh>

namespace oneflow {
namespace {
template <typename T>
__global__ void MyOpForwardGpu(const int n, const T *x, T *y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 3*x[i]*x[i]; }
}

class MyOpGpuFloatKernel final : public user_op::OpKernel {
public:
  MyOpGpuFloatKernel() = default;
  ~MyOpGpuFloatKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int32_t n = in_tensor->shape().elem_cnt();
    const float *in_ptr = in_tensor->dptr<float>();
    float *out_ptr = out_tensor->mut_dptr<float>();
    MyOpForwardGpu<float>
        <<<32, 1024, 0, ctx->device_ctx()->cuda_stream()>>>(n, in_ptr, out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MYOP_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("myop")                     \
      .SetCreateFn<MyOpGpuFloatKernel>()             \
      .SetIsMatchedHob(                              \
          (user_op::HobDeviceTag() == device) &     \
          (user_op::HobDataType("out", 0)            \
            == GetDataType<dtype>::value));

REGISTER_MYOP_KERNEL(DeviceType::kGPU, float)
REGISTER_MYOP_KERNEL(DeviceType::kGPU, double)
} // namespace
} // namespace oneflow

