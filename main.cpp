#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <memory>
#include <list>
#include <cmath>

namespace GML {
  enum COND {EQ, NEQ, LT, LTE, GT, GTE};
  enum MODE {BINARY, RANKED, MULTIPLE};

  template<typename T>
    class DATA : public std::vector<T> {
      public:
        DATA(std::vector<T>& r) : std::vector<T>::vector(r) {}
        DATA(std::vector<T>&& r) : std::vector<T>::vector(r) {}

        friend std::ostream& operator<<(std::ostream& out, const DATA& td) {
          out << "DATA({"; 

          for(size_t i = 0; i < td.size() - 1; ++i) {
            out << td[i] << ", ";
          }

          out << td.back() << "})";
          return out ;
        }

    };

  template<typename T>
    class TDATA : public DATA<T> {
      public:
        std::string label;

        TDATA(std::string data_label, std::vector<T>& r) : label{data_label}, DATA<T>(r) {}
        TDATA(std::string data_label, std::vector<T>&& r) : label{data_label}, DATA<T>(r) {}

        friend std::ostream& operator<<(std::ostream& out, const TDATA& td) {
          out << "TDATA(" << td.label << ", {"; 

          for(size_t i = 0; i < td.size() - 1; ++i) {
            out << td[i] << ", ";
          }

          out << td.back() << "})";
          return out ;
        }
    };

  template<typename T>
    struct TDATA_COL : public std::vector<TDATA<T>> {
      using std::vector<TDATA<T>>::vector;

      auto col_size() const {
        return (*this)[0].size();
      }

      std::unordered_map<std::string, size_t> count() const {
        std::unordered_map<std::string, size_t> data_counts{0};
        for(const auto tdata : *this) {
          const std::string& name = tdata.label;
          data_counts[name] += 1;
        }

        return data_counts;
      }

      friend std::ostream& operator<<(std::ostream& out, const TDATA_COL& tdcol) {
        out << "TDATA_COL(";
        size_t tdcol_size = tdcol.size();

        for(size_t i = 0; i < tdcol_size - 1; ++i) {
          out << tdcol[i] << ", ";
        }
        out << tdcol.back() << ")";
        return out;
      }

    };

  template<typename T>
    class QUESTION {
      protected:
        int _column;
        T _value;

      public:
        QUESTION() : _column{0}, _value{0} {}
        QUESTION(int column, T value) : _column{column}, _value{value} {}

        bool operator()(const DATA<T>& td, enum COND M = EQ) const {
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
        };

        friend std::ostream& operator<<(std::ostream& out, const QUESTION<T>& q) {
          out << "Question(" << q._column << ", " << q._value << ')'; 
          return out ;
        }
    };

  // IMPURITY DOES NOT HAVE DEFAULT VALUES!
  template<typename T>
    class DECISION_NODE {
      public:
        TDATA_COL<T> tdatacol;
        double impurity;
        QUESTION<T> question;
        std::shared_ptr<DECISION_NODE> true_branch;
        std::shared_ptr<DECISION_NODE> false_branch;

        DECISION_NODE() : impurity{0.0}, true_branch{nullptr}, false_branch{nullptr} {}
        DECISION_NODE(TDATA_COL<T>& tdc, double gini, QUESTION<T> q, std::shared_ptr<DECISION_NODE> tbranch = nullptr, std::shared_ptr<DECISION_NODE> fbranch = nullptr) 
          : tdatacol{tdc}, impurity{gini}, question{q}, true_branch{tbranch}, false_branch{fbranch} {}

        bool is_leaf() const {
          return (true_branch == nullptr) && (false_branch == nullptr);
        }

        friend std::ostream& operator<<(std::ostream& out, const DECISION_NODE& dnode) {
          out << "DECISION_NODE(" 
            << dnode.tdatacol
            << ", "
            << dnode.question 
            << ", " 
            << dnode.impurity 
            << ", " ;

          if(dnode.true_branch == nullptr)
            out << "NULL";
          else
            out << *(dnode.true_branch);

          out << ", ";

          if(dnode.false_branch == nullptr)
            out << "NULL";
          else
            out << *(dnode.false_branch);

          out << ")";
          return out;
        }
    };

  template<typename T>
    class TREE {
      private:
        const TDATA_COL<T>& _training_data;
        std::shared_ptr<DECISION_NODE<T>> _dtree;
        std::shared_ptr<DECISION_NODE<T>> _build_tree(TDATA_COL<T>& tdatacol);
        const DECISION_NODE<T>& _find_best_answer(const DATA<T>& data, const DECISION_NODE<T>& node) const {
          if(node.is_leaf()) 
            return node;

          if(node.question(data))
            return _find_best_answer(data, *node.true_branch);
          else
            return _find_best_answer(data, *node.false_branch);
        };

      public:
        TREE(TDATA_COL<T>& training_data);

        DECISION_NODE<T> predict(DATA<T> data) const {
          return _find_best_answer(data, *_dtree);
        }

        DECISION_NODE<T> true_branch() const {
          return *_dtree->true_branch;
        }

        DECISION_NODE<T> false_branch() const {
          return *_dtree->false_branch;
        }

        friend std::ostream& operator<<(std::ostream& out, const TREE& tree) {
          out << tree._dtree;
          return out;
        }
    };

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
    std::pair<double, QUESTION<T>> 
    find_best_split(const TDATA_COL<T>& tdatacol) {
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

  template<typename T> 
    std::shared_ptr<DECISION_NODE<T>> TREE<T>::_build_tree(TDATA_COL<T>& tdatacol) {
      auto [info_gain, question] = find_best_split(tdatacol);

      if(info_gain == 0) 
        return std::make_shared<DECISION_NODE<T>>(tdatacol, info_gain, question, nullptr, nullptr);

      auto [true_rows, false_rows] = partition(tdatacol, question);

      std::shared_ptr<DECISION_NODE<T>> true_branch = _build_tree(true_rows);
      std::shared_ptr<DECISION_NODE<T>> false_branch = _build_tree(false_rows);

      return std::make_shared<DECISION_NODE<T>>(tdatacol, info_gain, question, true_branch, false_branch);
    }

  template<typename T>
    TREE<T>::TREE(TDATA_COL<T>& training_data) : _training_data{training_data} {
      this->_dtree = this->_build_tree(training_data);
    }
}

using namespace std::literals::string_literals;

int main() {
  GML::DATA<std::string> data{{"Yellow"s, "Big"s}};

  GML::TDATA_COL<std::string> training_data({
      {"Apple"s, {"Green"s, "Big"s}},
      {"Apple"s, {"Yellow"s, "Big"s}},
      {"Lemon"s, {"Yellow"s, "Big"s}},
      {"Grape"s, {"Red"s, "Small"s}},
      {"Grape"s, {"Red"s, "Small"s}},
      });

  GML::TDATA_COL<std::string> no_mixing({
      {"Apple", {}},
      {"Orange", {}},
      {"Grape", {}},
      {"Lemon", {}},
      {"Blueberry", {}},
      });

  GML::TREE<std::string> tree(training_data);
  std::cout << tree.false_branch();

  return 0;
}
