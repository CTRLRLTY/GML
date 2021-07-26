#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "GML.hpp"

using namespace std::literals::string_literals;

GML::TDATA<std::string> tdata("Apple"s, {"Yellow"s, "Big"s});
GML::TDATA_COL<std::string> training_data({
    tdata,
    {"Apple"s, {"Green"s, "Big"s}},
    {"Grape"s, {"Red"s, "Small"s}},
    {"Grape"s, {"Red"s, "Small"s}},
    {"Lemon"s, {"Red"s, "Big"s}}
    });
GML::DATA<std::string> data({"Green", "Big"});

TEST_CASE("Testing DATA & TDATA Implementation") {
  CHECK(!data.empty());
  CHECK(!tdata.empty());
  CHECK(tdata.label.compare("Apple") == 0);
  CHECK(tdata.label.compare(training_data[0].label) == 0);
}

TEST_CASE("Testing TDATA_COL Implementation") {
  GML::CLASS_COUNT counts = training_data.count();

  CHECK(!training_data.empty());
  CHECK(training_data.size() == 5);
  CHECK(training_data[2].label.compare("Grape"s) == 0);
  CHECK(counts["Apple"] == 2);
  CHECK(counts["Grape"] == 2);
  CHECK(counts["Lemon"] == 1);
}

TEST_CASE("Testing TREE Implementation") {
  GML::TREE<std::string> tree1(training_data);
  auto nodedata1 = tree1.predict(data).nodedata();

  // Test wether tree was correctly constructed
  CHECK(!nodedata1.empty());
  CHECK(!tree1.empty()); 

  // Test wether tree was correctly move constructed
  auto tree2(std::move(tree1));
  auto nodedata2 = tree2.predict(data).nodedata();
  CHECK(!nodedata2.empty()); // Does the moved tree still works?
  CHECK(tree1.empty()); // Does the previous location still stores data?


  // Test wether tree was correctly copy constructed
  auto tree3(tree2);
  auto nodedata3 = tree2.predict(data).nodedata();
  CHECK(!nodedata3.empty()); // Does the moved tree still works?
  CHECK(!tree2.empty()); // Does the previous location still stores data?

  // Test tree copy assignment operation
  GML::TREE<std::string> tree4;
  tree4 = tree3;
  CHECK(!tree3.empty());

  // Test tree move assignment operation
  tree4 = std::move(tree3);
  CHECK(tree3.empty());
}
