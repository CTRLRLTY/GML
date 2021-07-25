#ifndef GML_H
#define GML_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>

namespace GML {
  enum COND {EQ, NEQ, LT, LTE, GT, GTE};
  enum MODE {BINARY, RANKED, MULTIPLE};

  using CLASS_COUNT = std::unordered_map<std::string, size_t>; // All classifier total amount inside a TDATA
  using PRES_CONFIDENCE = std::unordered_map<std::string, std::string>; // Prediction Result Confidence

  template<typename T>
    class DATA : public std::vector<T> {
      public:
        DATA(std::vector<T>& r);
        DATA(std::vector<T>&& r);

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

        TDATA(std::string data_label, std::vector<T>& r);
        TDATA(std::string data_label, std::vector<T>&& r);

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

      size_t col_size() const;
      CLASS_COUNT count() const;
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
        QUESTION();
        QUESTION(int column, T value);

        bool operator()(const DATA<T>& td, enum COND M = EQ) const;

        friend std::ostream& operator<<(std::ostream& out, const QUESTION<T>& q) {
          out << "Question(" << q._column << ", " << q._value << ')'; 
          return out ;
        }
    };


  template<typename T> struct NODE_DATA {
    double impurity;
    std::shared_ptr<TDATA_COL<T>> tdatacol_sptr;
    std::shared_ptr<CLASS_COUNT> count_sptr;
    std::shared_ptr<PRES_CONFIDENCE> confidence_sptr;

    NODE_DATA(
        double gini = 0.0,
        std::shared_ptr<TDATA_COL<T>> tdc_sptr = nullptr,
        std::shared_ptr<CLASS_COUNT> cnt_sptr = nullptr,
        std::shared_ptr<PRES_CONFIDENCE> cnf_sptr = nullptr
        );

    NODE_DATA(const NODE_DATA &nodedata);

    NODE_DATA(NODE_DATA &&nodedata);

    void operator=(const NODE_DATA& nodedata); 

    void operator=(NODE_DATA &&nodedata); 

    friend std::ostream& operator<<(std::ostream& out, const NODE_DATA& nodedata) {
      out << "NODE_DATA(" << nodedata.impurity << ", ";

      if(nodedata.tdatacol_sptr)
        out << *nodedata.tdatacol_sptr;
      else
        out << "nullptr";

      out << ", ";

      if(nodedata.count_sptr)
        out << nodedata.count_sptr;
      else
        out << "nullptr";

      out << ", ";

      if(nodedata.confidence_sptr)
        out << nodedata.confidence_sptr;
      else
        out << "nullptr";

      out << ")";
      return out;
    }
  };

  template<typename T>
    class DECISION_NODE {
      private: 
        std::shared_ptr<NODE_DATA<T>> _nodedata_sptr;
        std::shared_ptr<QUESTION<T>> _question_sptr;
        std::shared_ptr<DECISION_NODE> _true_branch_sptr;
        std::shared_ptr<DECISION_NODE> _false_branch_sptr;

      public:
        DECISION_NODE(
            std::shared_ptr<NODE_DATA<T>> nodedata_sptr = nullptr,
            std::shared_ptr<QUESTION<T>> question_sptr = nullptr,
            std::shared_ptr<DECISION_NODE> true_branch_sptr = nullptr,
            std::shared_ptr<DECISION_NODE> false_branch_sptr = nullptr
            );

        NODE_DATA<T> nodedata() const;
        QUESTION<T> question() const;
        DECISION_NODE true_branch() const;
        DECISION_NODE false_branch() const;
        bool is_leaf() const;

        friend std::ostream& operator<<(std::ostream& out, const DECISION_NODE& dnode) {
          out << "DECISION_NODE(" << *dnode._nodedata_sptr << ", ";

          if(dnode._question_sptr == nullptr)
            out << "nullptr";
          else
            out << *dnode._question_sptr;
          out << ", ";

          if(dnode._true_branch_sptr == nullptr)
            out << "nullptr";
          else
            out << *dnode._true_branch_sptr;
          out << ", ";

          if(dnode._false_branch_sptr == nullptr)
            out << "nullptr";
          else
            out << *dnode._false_branch_sptr;

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

        DECISION_NODE<T> _find_best_answer(const DATA<T>& data, const DECISION_NODE<T>& node) const;

      public:
        TREE(TDATA_COL<T>& training_data);

        DECISION_NODE<T> predict(DATA<T> data) const;

        friend std::ostream& operator<<(std::ostream& out, const TREE& tree) {
          out << *tree._dtree;
          return out;
        }
    };

  template<typename T>
    double gini(const TDATA_COL<T>& r);

  template<typename T>
    std::pair<TDATA_COL<T>, TDATA_COL<T>> partition(const TDATA_COL<T>& r, const QUESTION<T>& q);

  template<typename T>
    double info_gain(const TDATA_COL<T>& left, const TDATA_COL<T>& right, double base_impurity);

  template<typename T, enum MODE = BINARY>
    std::pair<double, QUESTION<T>> find_best_split(const TDATA_COL<T>& tdatacol);
}

#endif // GML_H
