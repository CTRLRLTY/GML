#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "GML.hpp"
#include <string>

using namespace std::literals::string_literals;

GML::TDATA<std::string> tdata("Apple"s, {"Yellow"s, "Big"s});
GML::TDATA_COL<std::string> training_data({
    tdata,
    {"Apple"s, {"Green"s, "Big"s}},
    {"Grape"s, {"Red"s, "Small"s}},
    {"Grape"s, {"Red"s, "Small"s}},
    {"Lemon"s, {"Red"s, "Big"s}}
    });
GML::TREE<std::string> tree(training_data);

TEST_CASE("Testing TDATA_COL Implementation") {
  GML::CLASS_COUNT counts = training_data.count();

  CHECK(tdata.label.compare("Apple") == 0);
  CHECK(tdata.label.compare(training_data[0].label) == 0);
  CHECK(training_data[2].label.compare("Grape"s) == 0);
  CHECK(counts["Apple"] == 2);
  CHECK(counts["Grape"] == 2);
  CHECK(counts["Lemon"] == 1);
}

TEST_CASE("Testing TREE Implementation") {
  GML::TREE<std::string> tree(training_data);
  GML::DATA<std::string> data({"Green", "Big"});
  auto nodedata = tree.predict(data).nodedata();
  auto counts = *nodedata.count_sptr;
  CHECK(counts["Apple"] == 2);
}
