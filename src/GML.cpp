#include "GML.h"

using namespace GML;

// DATA Definitions
template<typename T>
DATA<T>::DATA(std::vector<T>& r) : std::vector<T>::vector(r) {}
template<typename T>
DATA<T>::DATA(std::vector<T>&& r) : std::vector<T>::vector(r) {}

// TDATA Definitions
template<typename T>
TDATA<T>::TDATA(std::string data_label, std::vector<T>& r) : label{data_label}, DATA<T>(r) {}
template<typename T>
TDATA<T>::TDATA(std::string data_label, std::vector<T>&& r) : label{data_label}, DATA<T>(r) {}

// TDATA_COL Definitions
template<typename T>
size_t TDATA_COL<T>::col_size() const { return (*this)[0].size(); }
template<typename T>
CLASS_COUNT TDATA_COL<T>::count() const {
  CLASS_COUNT data_counts{0};

  for(const auto tdata : *this) {
    const std::string& name = tdata.label;
    data_counts[name] += 1;
  }

  return data_counts;
}

// QUESTION Definitions
template<typename T>
QUESTION<T>::QUESTION() : _column{0}, _value{0} {}
template<typename T>
QUESTION<T>::QUESTION(int column, T value) : _column{column}, _value{value} {}
template<typename T>
bool QUESTION<T>::operator()(const DATA<T>& td, enum COND M) const {
  T val = td[_column];
  switch(M) {
    case EQ:
      return _value == val;
    case NEQ:
      return _value != val;
    case LT:
      return _value < val;
    case LTE:
      return _value <= val;
    case GT:
      return _value > val;
    case GTE:
      return _value >= val;
  }
}

// NODE_DATA Definitions
template<typename T>
NODE_DATA<T>::NODE_DATA(
    double gini,
    std::shared_ptr<TDATA_COL<T>> tdc_sptr,
    std::shared_ptr<CLASS_COUNT> cnt_sptr,
    std::shared_ptr<PRES_CONFIDENCE> cnf_sptr
    ) : impurity{gini}, tdatacol_sptr{tdc_sptr}, count_sptr{cnt_sptr}, confidence_sptr{cnf_sptr} {}
template<typename T>
NODE_DATA<T>::NODE_DATA(const NODE_DATA &nodedata) : 
  impurity{nodedata.impurity}, 
  tdatacol_sptr{nodedata.tdatacol_sptr},
  count_sptr{nodedata.count_sptr},
  confidence_sptr{nodedata.confidence_sptr} 
{
  std::cout << "NODE_DATA COPY CONSTRUCTED" << std::endl; // DELETE LATER
}
template<typename T>
NODE_DATA<T>::NODE_DATA(NODE_DATA &&nodedata) : 
  impurity{nodedata.impurity}, 
  tdatacol_sptr{std::move(nodedata.tdatacol_sptr)},
  count_sptr{std::move(nodedata.count_sptr)},
  confidence_sptr{std::move(nodedata.confidence_sptr)} 
{
  std::cout << "NODE_DATA MOVE CONSTRUCTED" << std::endl; // DELETE LATER
  nodedata.impurity = 0.0;
}
template<typename T>
void NODE_DATA<T>::operator=(const NODE_DATA& nodedata) {
  std::cout << "NODE_DATA COPY ASSIGNED" << std::endl;
  impurity = nodedata.impurity;      
  tdatacol_sptr = nodedata.tdatacol_sptr;
  count_sptr = nodedata.count_sptr;
  confidence_sptr = nodedata.confidence_sptr;
}
template<typename T>
void NODE_DATA<T>::operator=(NODE_DATA &&nodedata) {
  std::cout << "NODE_DATA MOVE ASSIGNED" << std::endl;
  impurity = nodedata.impurity;
  tdatacol_sptr = std::move(nodedata.tdatacol_sptr);
  count_sptr = std::move(nodedata.count_sptr);
  confidence_sptr = std::move(nodedata.confidence_sptr);
  nodedata.impurity = 0.0;
}

// DECISION_NODE Definitions
template<typename T>
DECISION_NODE<T>::DECISION_NODE(
    std::shared_ptr<NODE_DATA<T>> nodedata_sptr,
    std::shared_ptr<QUESTION<T>> question_sptr,
    std::shared_ptr<DECISION_NODE> true_branch_sptr,
    std::shared_ptr<DECISION_NODE> false_branch_sptr
    ) : _nodedata_sptr{nodedata_sptr}, _question_sptr{question_sptr}, _true_branch_sptr{true_branch_sptr}, _false_branch_sptr{false_branch_sptr} {}
template<typename T>
NODE_DATA<T> DECISION_NODE<T>::nodedata() const { return *_nodedata_sptr; }
template<typename T>
QUESTION<T> DECISION_NODE<T>::question() const { return *_question_sptr; }
template<typename T>
DECISION_NODE<T> DECISION_NODE<T>::true_branch() const { return *_true_branch_sptr; }
template<typename T>
DECISION_NODE<T> DECISION_NODE<T>::false_branch() const { return *_false_branch_sptr; }
template<typename T>
bool DECISION_NODE<T>::is_leaf() const { return !_true_branch_sptr && !_false_branch_sptr; }


// TREE Definitions
template<typename T>
TREE<T>::TREE(TDATA_COL<T>& training_data) : _training_data{training_data} {
  this->_dtree = this->_build_tree(training_data);
}

template<typename T> 
std::shared_ptr<DECISION_NODE<T>> TREE<T>::_build_tree(TDATA_COL<T>& tdatacol) {
  auto [info_gain, question] = find_best_split(tdatacol);

  NODE_DATA<T> nodedata( 
      info_gain, 
      std::make_shared<TDATA_COL<T>>(tdatacol),
      std::make_shared<CLASS_COUNT>(tdatacol.count())
      );

  if(info_gain == 0) 
    return std::make_shared<DECISION_NODE<T>>(std::make_shared<NODE_DATA<T>>(std::move(nodedata)));

  auto [true_rows, false_rows] = partition<T>(tdatacol, question);

  std::shared_ptr<DECISION_NODE<T>> true_branch = _build_tree(true_rows);
  std::shared_ptr<DECISION_NODE<T>> false_branch = _build_tree(false_rows);

  return std::make_shared<DECISION_NODE<T>>(
      std::make_shared<NODE_DATA<T>>(std::move(nodedata)), 
      std::make_shared<QUESTION<T>>(question), true_branch, false_branch
      );
}

template<typename T>
DECISION_NODE<T> TREE<T>:: predict(DATA<T> data) const {
  return _find_best_answer(data, *_dtree);
}

template<typename T>
DECISION_NODE<T> TREE<T>::_find_best_answer(const DATA<T>& data, const DECISION_NODE<T>& node) const {
  if(node.is_leaf()) 
    return node;

  if(node.question()(data)) 
    return _find_best_answer(data, node.true_branch());
  else 
    return _find_best_answer(data, node.false_branch());
}


// Function Definitions
template<typename T>
double gini(const TDATA_COL<T>& r) {
  auto counts = r.count();
  double impurity = 1.0;
  for(const auto& [_name, amount] : counts) {
    double correct_label_probability = amount / ((double) r.size()); 
    impurity -= pow(correct_label_probability, 2.0);
  }

  return impurity;
}

template<typename T>
std::pair<TDATA_COL<T>, TDATA_COL<T>> partition(const TDATA_COL<T>& r, const QUESTION<T>& q) {
  TDATA_COL<T> true_rows, false_rows;

  for(const auto& x : r) {
    if(q(x))
      true_rows.push_back(x);
    else
      false_rows.push_back(x);
  }
  return {true_rows, false_rows};
}

template<typename T>
double info_gain(const TDATA_COL<T>& left, const TDATA_COL<T>& right, double base_impurity) {
  int left_size = left.size();
  double item_ratio = ((double) left_size) / (left_size + right.size());

  return base_impurity - item_ratio * gini(left) - (1 - item_ratio) * gini(right);
};

template<typename T, enum MODE = BINARY>
std::pair<double, QUESTION<T>> find_best_split(const TDATA_COL<T>& tdatacol) {
  double best_gain = 0.0,
         root_impurity = gini(tdatacol);
  int column_size = tdatacol.col_size();

  QUESTION<T> best_question; 

  for(int column_idx = 0; column_idx < column_size; ++column_idx) {
    for(const auto& tdata : tdatacol) {
      double gain {0.0};
      QUESTION<T> q{column_idx, tdata[0]};
      auto [true_rows, false_rows] = partition<T>(tdatacol, q);

      if(true_rows.size() == 0 || false_rows.size() == 0)
        continue;

      gain = info_gain(true_rows, false_rows, root_impurity);
      if(best_gain <= gain) {
        best_gain = gain;
        best_question = q;
      }
    }
  }

  return {best_gain, best_question};
}


