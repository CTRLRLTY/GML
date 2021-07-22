#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <cmath>

namespace GML {
  enum COND {EQ, NEQ, LT, LTE, GT, GTE};
  enum MODE {BINARY, RANKED, MULTIPLE};

  template<typename T>
  class TDATA : public std::vector<T> {
    public:
      std::string label;

      TDATA(std::string data_label, std::vector<std::string>& r) : label{data_label}, std::vector<std::string>::vector(r) {}
      TDATA(std::string data_label, std::vector<std::string>&& r) : label{data_label}, std::vector<std::string>::vector(r) {}

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

        bool operator()(const TDATA<T>& td, enum COND M = EQ) const {
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
    void print_vector(const std::vector<T>& v) {
      for(const auto& content : v) {
        std::cout << content << std::endl;
      }
    }
}

using namespace std::literals::string_literals;

int main() {
  GML::TDATA<std::string> data{"Apple", {"Red", "Big"}};

  GML::TDATA_COL<std::string> training_data({
      {"Apple", {"Green", "Big"}},
      {"Apple", {"Yellow", "Big"}},
      {"Lemon", {"Yellow", "Big"}},
      {"Grape", {"Red", "Small"}},
      {"Grape", {"Red", "Small"}},
      });

  GML::TDATA_COL<std::string> no_mixing({
      {"Apple", {}},
      {"Orange", {}},
      {"Grape", {}},
      {"Lemon", {}},
      {"Blueberry", {}},
      });

  auto [bg, bq] = GML::find_best_split<std::string>(training_data);
  std::cout << bg << " " << bq;

  return 0;
}
