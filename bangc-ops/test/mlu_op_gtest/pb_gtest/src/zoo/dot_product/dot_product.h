#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_DOT_PRODUCT_DOT_PRODUCT_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_DOT_PRODUCT_DOT_PRODUCT_H_

#include "executor.h"

namespace mluoptest {

class DotProductExecutor : public Executor {
  public:
    DotProductExecutor() {}
    ~DotProductExecutor() {}

    void paramCheck();
    void compute();
    void cpuCompute();
    int64_t getTheoryOps() override;
};

}

#endif