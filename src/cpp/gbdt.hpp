// Author: qiyiping@gmail.com (Yiping Qi)
#ifndef _GBDT_H_
#define _GBDT_H_
#include "tree.hpp"

namespace gbdt {
class GBDT {
 public:
  GBDT(): trees(NULL),
          bias(0),
          shrinkage(g_conf.shrinkage),
          iterations(g_conf.iterations),
          gain(NULL), 
          gain_predict(NULL){}

  void Fit(DataVector *d);
  ValueType Predict(const Tuple &t)  {
    return Predict(t, iterations, true);
  }

  std::string Save() const;
  void Load(const std::string &s);

  double *GetGain() { return gain; }

  ~GBDT();
 private:
  ValueType Predict(const Tuple &t, size_t n, bool flag);
  void Init(const DataVector &d, size_t len);
 private:
  RegressionTree *trees;
  ValueType bias;
  ValueType shrinkage;
  size_t iterations;

  double *gain;
  // the predict feature gain
  double * gain_predict;

  DISALLOW_COPY_AND_ASSIGN(GBDT);
};
}

#endif /* _GBDT_H_ */
