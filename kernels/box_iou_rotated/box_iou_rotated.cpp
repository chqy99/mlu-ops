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
#include "box_iou_rotated.h"
#include <algorithm>
#include <string>

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

// each box data contains 5 number: x, y, w, h, a
#define SINGLE_BOX_DIM 5
#define MAX_BOX_NUM 10000000

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const bool aligned,
                       const int num_box1, const int num_box2) {
  // When current MLU arch only support Block type job
  if (mluop::runtime::getJobLimitCapability(handle) == CN_KERNEL_CLASS_BLOCK) {
    *k_type = cnrtFuncTypeBlock;
    k_dim->x = 1;
    k_dim->y = 1;
    k_dim->z = 1;
    VLOG(5) << "Launch Kernel MLUKernelBoxIouRotated in BLOCK type";
    return;
  }

  // union1 policy func
  *k_type = cnrtFuncTypeUnion1;
  // dimx equals to num of mlu cores in each cluster
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  // dimy equals to num of current available clusters
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;

  // if total_num < 64, use only one mlu core;
  const uint32_t single_core_small_case = 64;

  if (single_core_small_case >= num_box1) {  // only 1 mlu core enough
    *k_type = cnrtFuncTypeBlock;
    k_dim->x = 1;
    k_dim->y = 1;
    VLOG(5) << "Launch Kernel MLUKernelBoxIouRotated in BLOCK type";
    return;
  }
  // aligned = false, always partition on num_box1, when >
  // single_core_small_case
  if (aligned) {
    if (k_dim->x * single_core_small_case >= num_box1) {  // 1 cluster
      k_dim->y = 1;
    } else {
      // use how many clusters to start Union1 job
      uint32_t use_cluster_num = num_box1 / (k_dim->x * single_core_small_case);
      k_dim->y = std::min(k_dim->y, use_cluster_num);
    }
  }
  VLOG(5) << "Launch Kernel MLUKernelBoxIouRotated in UNION1 type";
}

mluOpStatus_t MLUOP_WIN_API
mluOpBoxIouRotated(mluOpHandle_t handle, const int mode, const bool aligned,
                   const mluOpTensorDescriptor_t box1_desc, const void *box1,
                   const mluOpTensorDescriptor_t box2_desc, const void *box2,
                   const mluOpTensorDescriptor_t ious_desc, void *ious) {
  // desc null pointer check
  PARAM_CHECK("[mluOpBoxIouRotated]", handle != NULL);
  PARAM_CHECK("[mluOpBoxIouRotated]", box1_desc != NULL);
  PARAM_CHECK("[mluOpBoxIouRotated]", box2_desc != NULL);
  PARAM_CHECK("[mluOpBoxIouRotated]", ious_desc != NULL);

  // datatype check
  PARAM_CHECK("[mluOpBoxIouRotated]",
              box1_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK_EQ("[mluOpBoxIouRotated]", box1_desc->getDtype(),
                 box2_desc->getDtype());
  PARAM_CHECK_EQ("[mluOpBoxIouRotated]", box1_desc->getDtype(),
                 ious_desc->getDtype());

  // param check
  if (mode != 0 && mode != 1) {
    LOG(ERROR)
        << "[mluOpBoxIouRotated]: mode should set to 0(IOU) or 1(IOF), but "
        << mode << " found.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // dims and shape check
  PARAM_CHECK_EQ("[mluOpBoxIouRotated]", box1_desc->getDim(), 2);
  PARAM_CHECK_EQ("[mluOpBoxIouRotated]", box2_desc->getDim(), 2);
  if (box1_desc->getDimIndex(box1_desc->getDim() - 1) != SINGLE_BOX_DIM &&
      box1_desc->getDimIndex(0) != 0) {
    LOG(ERROR)
        << "[mluOpBoxIouRotated] Check failed: The Boxes' last dimenstion "
           "should be 5 or "
        << "the first dimension should be 0. But now box1's last dimension is "
        << box1_desc->getDimIndex(box1_desc->getDim() - 1)
        << ", box1's first dimension is " << box1_desc->getDimIndex(0) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (box2_desc->getDimIndex(box2_desc->getDim() - 1) != SINGLE_BOX_DIM &&
      box2_desc->getDimIndex(0) != 0) {
    LOG(ERROR)
        << "[mluOpBoxIouRotated] Check failed: The Boxes' last dimenstion "
           "should be 5 or "
        << "the first dimension should be 0. But now box2's last dimension is "
        << box2_desc->getDimIndex(box2_desc->getDim() - 1)
        << ", box2's first dimension is " << box2_desc->getDimIndex(0) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (ious_desc->getDimIndex(0) != box1_desc->getDimIndex(0)) {
    LOG(ERROR)
        << "[mluOpBoxIouRotated] Check failed: Whether it is aligned or not,"
        << "ious_desc->getDimIndex(0) should equal to "
           "box1_desc->getDimIndex(0). But now "
        << "ious_desc->getDimIndex(0) is " << ious_desc->getDimIndex(0)
        << ", box1_desc->getDimIndex(0) is " << box1_desc->getDimIndex(0)
        << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (aligned) {
    if (ious_desc->getDim() != 1) {
      LOG(ERROR) << "[mluOpBoxIouRotated] Check failed: If it is aligned mode, "
                 << "ious_desc->getDim() should equal to 1. But now is "
                 << ious_desc->getDim() << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (box1_desc->getDimIndex(0) != box2_desc->getDimIndex(0)) {
      LOG(ERROR) << "[mluOpBoxIouRotated] Check failed: If it is aligned mode, "
                 << "box1_desc->getDimIndex(0) should equal to "
                    "box2_desc->getDimIndex(0). But now "
                 << "box1_desc->getDimIndex(0) is " << box1_desc->getDimIndex(0)
                 << ", box2_desc->getDimIndex(0) is "
                 << box2_desc->getDimIndex(0) << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (handle->arch < 592 && box1_desc->getDimIndex(0) > MAX_BOX_NUM) {
      LOG(ERROR)
          << "[mluOpBoxIouRotated] Check failed: If it is aligned mode, "
          << "on MLU300 box1_desc->getDimIndex(0) should less than or equal to "
          << "10,000,000 . But now is " << box1_desc->getDimIndex(0) << ".";
      return MLUOP_STATUS_NOT_SUPPORTED;
    }
  } else {
    if (ious_desc->getDim() != 2) {
      LOG(ERROR)
          << "[mluOpBoxIouRotated] Check failed: If it is non-aligned mode, "
          << "ious_desc->getDim() should equal to 2. But now is "
          << ious_desc->getDim() << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (ious_desc->getDimIndex(1) != box2_desc->getDimIndex(0)) {
      LOG(ERROR)
          << "[mluOpBoxIouRotated] Check failed: If it is non-aligned mode, "
          << "ious_desc's last dim should equal to box2_desc's first dim "
          << box2_desc->getDimIndex(0) << ", But now ious_desc's last dim is "
          << ious_desc->getDimIndex(1) << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // stride check
  STRIDE_TENSOR_CHECK("[mluOpBoxIouRotated]:", box1_desc,
                      "box1_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpBoxIouRotated]:", box2_desc,
                      "box2_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpBoxIouRotated]:", ious_desc,
                      "ious_desc must be contiguous");

  // 0-element check, after dim and shape check
  if (box1_desc->getDimIndex(0) * box2_desc->getDimIndex(0) == 0) {
    VLOG(5) << "[mluOpBoxIouRotated] Skip zero element boxes.";
    return MLUOP_STATUS_SUCCESS;
  }

  const size_t box1_element_num = mluOpGetTensorElementNum(box1_desc);
  const size_t box2_element_num = mluOpGetTensorElementNum(box2_desc);
  const size_t ious_element_num = mluOpGetTensorElementNum(ious_desc);

  // check large tensor
  TENSOR_NUM_CHECK("[mluOpBoxIouRotated]", box1_element_num, LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK("[mluOpBoxIouRotated]", box2_element_num, LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK("[mluOpBoxIouRotated]", ious_element_num, LARGE_TENSOR_NUM,
                   "");

  // data nullptr should check after 0-element check
  PARAM_CHECK("[mluOpBoxIouRotated]", box1 != NULL);
  PARAM_CHECK("[mluOpBoxIouRotated]", box2 != NULL);
  PARAM_CHECK("[mluOpBoxIouRotated]", ious != NULL);

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("box_iou_rotated", "BOX_IOU_ROTATED");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "input", box1, box1_desc);
    GEN_CASE_DATA_REAL(true, "input", box2, box2_desc);
    GEN_CASE_DATA_REAL(false, "output", ious, ious_desc);
    GEN_CASE_OP_PARAM_SINGLE(0, "box_iou_rotated", "mode", mode);
    GEN_CASE_OP_PARAM_SINGLE(2, "box_iou_rotated", "aligned", aligned);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 3e-3, 3e-3, 0);
  }

  int32_t num_box1 = box1_element_num / SINGLE_BOX_DIM;
  int32_t num_box2 = box2_element_num / SINGLE_BOX_DIM;

  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, aligned, num_box1, num_box2);

  VLOG(5) << "[mluOpBoxIouRotated] launch kernel policyFunc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "].";
  CHECK_RETURN("[mluOpBoxIouRotated]",
               (KernelBoxIouRotated(k_dim, k_type, handle->queue,
                                    box1_desc->getDtype(), box1, box2, ious,
                                    num_box1, num_box2, mode, aligned)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
