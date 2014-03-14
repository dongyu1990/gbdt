// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _TREE_H_
#define _TREE_H_

#include "data.hpp"
#include <map>
#include <stack>
#include <cmath>
#include <iostream>
#include <vector>

namespace gbdt {
class Node {
 public:
  enum {LT, GE, UNKNOWN, CHILDSIZE};

  Node() {
    child[LT] = NULL;
    child[GE] = NULL;
    child[UNKNOWN] = NULL;
    index = -1;
    value = 0;
    leaf = false;
    pred = 0;
  }

  ~Node() {
    delete child[LT];
    delete child[GE];
    delete child[UNKNOWN];
  }

  static void Fit(DataVector *data,
                  Node *node,
                  int depth);

  static ValueType Predict(Node *root, const Tuple &t);

  Node *child[CHILDSIZE];
  int index;
  ValueType value;
  bool leaf;
  ValueType pred;
 private:
  DISALLOW_COPY_AND_ASSIGN(Node);
};

class RegressionTree {
 public:
  RegressionTree(): root(NULL), gain(NULL),gain_predict(NULL) {}
  ~RegressionTree() { delete root; delete[] gain; }

  void Fit(DataVector *data) { Fit(data, data->size()); }
  void Fit(DataVector *data, size_t len);

  //ValueType Predict(const Tuple &t, bool flag);
  ValueType Predict(const Tuple &t, bool flag, int num);

  std::string Save() const;
  void Load(const std::string &s);

  double *GetGain() { return gain; }
  double *GetGain_predict() { return gain_predict; }

 private:
  static void Fit(DataVector *data,
                  Node *node,
                  size_t depth,
                  double *gain) { Fit(data, data->size(), node, depth, gain); }

  static void Fit(DataVector *data,
                  size_t len,
                  Node *node,
                  size_t depth,
                  double *gain);

  static ValueType Predict(const Node *node, const Tuple &t, int num, double * gain_predict);
  static ValueType Predict(const Node *node, const Tuple &t);

  static void SaveAux(const Node *node,
                      std::vector<const Node *> *nodes,
                      std::map<const void *, size_t> *position_map);

 private:
  Node *root;
  double *gain;
  double *gain_predict;

  DISALLOW_COPY_AND_ASSIGN(RegressionTree);
};

}

#endif /* _TREE_H_ */
