// Author: qiyiping@gmail.com (Yiping Qi)
#include "gbdt.hpp"
#include "fitness.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <boost/lexical_cast.hpp>
#include "time.hpp"
#include "auc.hpp"

#include <cstdlib>
#include <ctime>

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace gbdt;

int main(int argc, char *argv[]) {
  std::srand ( unsigned ( std::time(0) ) );

#ifdef USE_OPENMP
  const int threads_wanted = 4;
  omp_set_num_threads(threads_wanted);
#endif

  g_conf.number_of_feature = 3;
  g_conf.max_depth = 4;
  g_conf.iterations = 10;
  g_conf.shrinkage = 0.1F;

  if (argc < 3) return -1;
    std::string model_file(argv[1]);
    std::string test_file(argv[2]);
  //std::string train_file(argv[1]);
  //std::string test_file(argv[2]);

  //if (argc > 3) {
  //  g_conf.max_depth = boost::lexical_cast<size_t>(argv[3]);
  //}

  //if (argc > 4) {
  //  g_conf.iterations = boost::lexical_cast<size_t>(argv[4]);
  //}

  //if (argc > 5) {
  //  g_conf.shrinkage = boost::lexical_cast<float>(argv[5]);
  //}

  //if (argc > 6) {
  //  g_conf.feature_sample_ratio = boost::lexical_cast<float>(argv[6]);
  //}

  //if (argc > 7) {
  //  g_conf.data_sample_ratio = boost::lexical_cast<float>(argv[7]);
  //}

  //int debug = 0;
  //if (argc > 8) {
  //  debug = boost::lexical_cast<int>(argv[8]);
  //}

  //g_conf.loss = SQUARED_ERROR;
  //g_conf.debug = debug > 0? true : false;

  //DataVector d;
  //bool r = LoadDataFromFile(train_file, &d);
  //assert(r);

  //g_conf.min_leaf_size = d.size() * g_conf.data_sample_ratio / 40;
  //std::cout << "configure: " << std::endl
  //          << g_conf.ToString() << std::endl;

  //if (argc > 9) {
  //  g_conf.LoadFeatureCost(argv[9]);
  //}

  //if (argc > 10) {
  //  g_conf.number_of_feature = boost::lexical_cast<size_t>(argv[10]);
  //}
  std::ifstream fin(model_file.c_str());
  std::string s;
  std::string all;
  while(getline(fin,s))
  {
    all += s;
    all += '\n';
  }
  //std::cout<<all<<std::endl;
  GBDT gbdt;
  Elapsed elapsed;
  gbdt.Load(all);
  //gbdt.Fit(&d);
  //std::cout << "fit time: " << elapsed.Tell() << std::endl;

  //std::string model_file = train_file + ".model";
  //std::ofstream model_output(model_file.c_str());
  //model_output << gbdt.Save();

  //CleanDataVector(&d);
  //FreeVector(&d);

  DataVector d2;
  bool r = LoadDataFromFile(test_file, &d2);
  assert(r);

  //elapsed.Reset();
  DataVector::iterator iter = d2.begin();
  PredictVector predict;
  for ( ; iter != d2.end(); ++iter) {
    ValueType p = Logit(gbdt.Predict(**iter,2));
    //predict.push_back(p);

  }
  //std::cout << "predict time: " << elapsed.Tell() << std::endl;
  return 0;
}
