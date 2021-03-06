// Author: qiyiping@gmail.com (Yiping Qi)

#include "gbdt.hpp"
#include "fitness.hpp"
#include <fstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <boost/lexical_cast.hpp>

using namespace gbdt;

int main(int argc, char *argv[]) {
  std::string model;
  std::ifstream stream(argv[1]);
  assert(stream);

  stream.seekg(0, std::ios::end);
  model.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);
  model.assign(std::istreambuf_iterator<char>(stream),
               std::istreambuf_iterator<char>());

  GBDT gbdt;
  gbdt.Load(model);

  size_t feature_num = boost::lexical_cast<size_t>(argv[2]);

  g_conf.number_of_feature = feature_num;

  DataVector d;
  LoadDataFromFile(argv[3], &d);

  Loss loss_type = SQUARED_ERROR;
  if (argc > 4 && std::strcmp(argv[4], "logit") == 0) {
    loss_type = LOG_LIKELIHOOD;
  }

  g_conf.loss = loss_type;

  DataVector::iterator iter = d.begin();
  PredictVector predict;
  double *x = new double[feature_num];
  for ( ; iter != d.end(); ++iter) {
    ValueType p;
    for (size_t i = 0; i < feature_num; ++i)
      x[i] = 0.0;

    if (loss_type == SQUARED_ERROR) {
      p = gbdt.Predict(**iter, x);
      predict.push_back(p);
    } else if (loss_type == LOG_LIKELIHOOD) {
      p = gbdt.Predict(**iter, x);
      p = Logit(p);
      predict.push_back(p);
    }

    std::cout << "tuple: " << (*iter)->ToString() << std::endl
              << "predict: " << p << std::endl;

    std::cout << "x: ";
    for (size_t i = 0; i < feature_num; ++i) {
      std::cout << i << ":" << x[i] << " ";
    }
    std::cout << std::endl;

  }
  return 0;
}
