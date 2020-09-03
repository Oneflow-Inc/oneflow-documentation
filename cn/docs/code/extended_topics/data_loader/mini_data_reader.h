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
#ifndef ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_
#define ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_

#include "oneflow/user/data/data_reader.h"
#include "oneflow/user/data/random_shuffle_dataset.h"
#include "oneflow/user/data/batch_dataset.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/user/data/parser.h"
#include <cstddef>
#include <cstdint>
#include <iostream>


namespace oneflow {
namespace data {
  using namespace std;

class MiniDataset final : public Dataset<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;

  MiniDataset(user_op::KernelInitContext* ctx) {
    current_epoch_ = 0;
    shuffle_after_epoch_ = ctx->Attr<bool>("shuffle_after_epoch");

    // in stream
    data_part_num_ = ctx->Attr<int32_t>("data_part_num");
    std::string data_dir = ctx->Attr<std::string>("data_dir");
    std::string part_name_prefix = ctx->Attr<std::string>("part_name_prefix");
    int32_t part_name_suffix_length = ctx->Attr<int32_t>("part_name_suffix_length");

    for (int i = 0; i < data_part_num_; ++i) {
      std::string num = std::to_string(i);
      int32_t zero_count =
          std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
      data_file_paths_.push_back(
          JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
    }

    parallel_id_ = ctx->parallel_ctx().parallel_id();
    parallel_num_ = ctx->parallel_ctx().parallel_num();
    CHECK_LE(parallel_num_, data_part_num_);
    BalancedSplitter bs(data_part_num_, parallel_num_);
    range_ = bs.At(parallel_id_);
    std::vector<std::string> local_file_paths = GetLocalFilePaths();
    save_to_local_ = Global<const IOConf>::Get()->save_downloaded_file_to_local_fs();
    in_stream_.reset(
        new PersistentInStream(DataFS(), local_file_paths, !shuffle_after_epoch_, save_to_local_));
  }
  ~MiniDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    LoadTargetPtr sample_ptr(new TensorBuffer());

    std::string sampleline;
    if (in_stream_->ReadLine(&sampleline) != 0) {
      ShuffleAfterEpoch();
      in_stream_->ReadLine(&sampleline);
    }

    auto numbers = commaSplit(sampleline);
    sample_ptr->Resize(Shape({2}), DataType::kDouble);
    auto pNums = sample_ptr->mut_data<double>();
    pNums[0] = std::stod(numbers[0]);
    pNums[1] = std::stod(numbers[1]); 
    ret.push_back(std::move(sample_ptr));

    return ret;
  }

 private:
  vector<string> commaSplit(string strtem)
  {
      vector<string> strvec;

      string::size_type pos1, pos2;
      pos2 = strtem.find(',');
      pos1 = 0;
      while (string::npos != pos2)
      {
          strvec.push_back(strtem.substr(pos1, pos2 - pos1));

          pos1 = pos2 + 1;
          pos2 = strtem.find(',', pos1);
      }
      strvec.push_back(strtem.substr(pos1));
      return strvec;
  }

  void ShuffleAfterEpoch() {
    CHECK(shuffle_after_epoch_);
    current_epoch_++;  // move to next epoch
    std::mt19937 g(kOneflowDatasetSeed + current_epoch_);
    std::shuffle(data_file_paths_.begin(), data_file_paths_.end(), g);
    std::vector<std::string> local_file_paths = GetLocalFilePaths();
    in_stream_.reset(new PersistentInStream(DataFS(), local_file_paths, false, save_to_local_));
  }

  std::vector<std::string> GetLocalFilePaths() {
    std::vector<std::string> ret;
    for (int i = range_.begin(); i < range_.end(); ++i) { ret.push_back(data_file_paths_.at(i)); }
    return ret;
  }

  int32_t current_epoch_;
  bool shuffle_after_epoch_;

  int32_t data_part_num_;
  int32_t parallel_id_;
  int32_t parallel_num_;
  Range range_;
  std::vector<std::string> data_file_paths_;
  bool save_to_local_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

class MiniParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  MiniParser() = default;
  ~MiniParser() = default;

  void Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
             user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    double* dptr = out_tensor->mut_dptr<double>();
    
    MultiThreadLoop(batch_data->size(), [&](size_t i) {
      TensorBuffer* buffer = batch_data->at(i).get();
      dptr[i*2]= *(buffer->data<double>());
      dptr[i*2+1]= *(buffer->data<double>()+1);
    });
    
    if (batch_data->size() != out_tensor->shape().elem_cnt()) {
      CHECK_EQ(out_tensor->mut_shape()->NumAxes(), 1);
      out_tensor->mut_shape()->Set(0, batch_data->size());
    }
  }
};

class MiniDataReader final : public DataReader<TensorBuffer> {
 public:
  MiniDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new MiniDataset(ctx));
    parser_.reset(new MiniParser());

    int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
  ~MiniDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_
