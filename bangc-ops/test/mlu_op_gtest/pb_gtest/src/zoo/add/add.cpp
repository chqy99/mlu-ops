/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "add.h"

#include <iostream>

#include "mlu_op.h"

namespace mluoptest {

// 这个文件里有三个函数是必须要有的
// 第一个是:算子名字_executor::param_check(), 参数检查，顾名思义
void AddExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "[AddExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[AddExecutor] output number is wrong. ");
}

// 第二个是：算子名字_executor::compute(), 是创建算子MLU运算的函数
void AddExecutor::compute() {
  // 从tensor_desc_[].tensor中获取input或output的desc描述参数。
  // 从data_vector_[].device_ptr中依次获取input或output数据。它们都是指向ddr的指针。
  auto x_desc = tensor_desc_[0].tensor;
  auto x_ptr = data_vector_[0].device_ptr;
  auto y_desc = tensor_desc_[1].tensor;
  auto y_ptr = data_vector_[1].device_ptr;

  // 从测例的param模块获取alpha参数
  alpha_ = parser_->getProtoNode()->add_param().alpha();

  auto output_desc = tensor_desc_[2].tensor;
  auto output = data_vector_[2].device_ptr;

  // lunch kernel
  VLOG(4) << "[AddExecutor] call mluOpAdd()";

  /* interface_timer_.start()和interface_timer_.stop()是时间戳，
     用于计算 算子api的执行时间。*/
  interface_timer_.start();
  // 这一步调用mluOpAdd算子的api，将input、output和参数放入对应位置。
  MLUOP_CHECK(mluOpAdd(handle_, x_desc, x_ptr, y_desc, y_ptr,
                       alpha_, output_desc, output));
  interface_timer_.stop();

  data_vector_[2].is_output = true;
}

// 第三个是：算子名字_executor::cpuCompute(), 是创建算子cpu运算的函数
void AddExecutor::cpuCompute() {
  VLOG(4) << "[AddExecutor] call cpuCompute()";

  // 可以通过parser_>getInputDataCount()函数直接获取tensor的元素个数
  auto elem_num = parser_->getInputDataCount(0);

  // cpu对应的input数据指针是cpu_fp32_input_[]，output是cpu_fp32_output_[]
  for (int i = 0; i < elem_num; ++i) {
    cpu_fp32_output_[0][i] = alpha_ + cpu_fp32_input_[0][i]
                            + cpu_fp32_input_[1][i];

    // 可以使用VLOG进行打印，见最后一小节
    // 也可以使用std::cout进行打印
    std::cout << "output[ " << i << "] = " << cpu_fp32_output_[0][i] << " = "
              << alpha_ << " + "
              << cpu_fp32_input_[0][i] << " + "
              << cpu_fp32_input_[1][i] << std::endl;
  }
}

// 算子理论计算量
int64_t AddExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getInputDataCount(0) * 2;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
