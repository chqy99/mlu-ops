#include "dot_product.h"

#include <iostream>

#include "mlu_op.h"

namespace mluoptest {
  void DotProductExecutor::paramCheck() {
    GTEST_CHECK(parser_->inputs().size() == 2, "[DotProductExecutor] input number is wrong. ");
    GTEST_CHECK(parser_->outputs().size() == 1, "[DotProductExecutor] output number is wrong. ");
  }

  void DotProductExecutor::compute() {
    auto x_desc = tensor_desc_[0].tensor;
    auto x_ptr = data_vector_[0].device_ptr;
    auto y_desc = tensor_desc_[1].tensor;
    auto y_ptr = data_vector_[1].device_ptr;

    auto output_desc = tensor_desc_[2].tensor;
    auto output = data_vector_[2].device_ptr;
    
    VLOG(4) << "[DotProductExecutor] call mluOpDotProduct";
    interface_timer_.start();
     
    MLUOP_CHECK(mluOpDotProduct(handle_, x_desc, x_ptr, y_desc, y_ptr, output_desc, output));
    interface_timer_.stop();

    data_vector_[2].is_output = true;
  }

  void DotProductExecutor::cpuCompute() {
    VLOG(4) << "[DotProductExecutor] call cpuCompute()";

    auto elem_num = parser_->getInputDataCount(0);

    for (int i = 0; i < elem_num; ++i) {
      cpu_fp32_output_[0][0] += cpu_fp32_input_[0][i] * cpu_fp32_input_[1][i];

      // std::cout << "output[ " << i << "] = " << cpu_fp32_output_[0][0] << " += "
      //           << cpu_fp32_input_[0][i] << " * "
      //           << cpu_fp32_input_[1][i] << std::endl;
    }
  }

  int64_t DotProductExecutor::getTheoryOps() {
    int64_t theory_ops = parser_->getInputDataCount(0) * 2;
    VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
    return theory_ops;
  }
}