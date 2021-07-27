#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "GML.hpp"

using namespace std::literals::string_literals;

// Base Sets
GML::DATA<std::string> data({"Green", "Big"});
GML::TDATA<std::string> tdata("Apple"s, {"Yellow"s, "Big"s});
GML::TDATA_COL<std::string> training_data({
    tdata,
    {"Apple"s, {"Green"s, "Big"s}},
    {"Grape"s, {"Red"s, "Small"s}},
    {"Grape"s, {"Red"s, "Small"s}},
    {"Lemon"s, {"Yellow"s, "Big"s}}
    });

TEST_CASE("Testing DATA & TDATA Implementation") {
  REQUIRE(!data.empty());
  REQUIRE(!tdata.empty());
  REQUIRE(tdata[0].compare("Yellow"s) == 0);
  REQUIRE(tdata[1].compare("Big"s) == 0);
  REQUIRE(tdata.label.compare("Apple") == 0);


  GML::TDATA<std::string> tdata1;
  CHECK(tdata1.empty()); // Test If we could create an empty tdata
  tdata1.push_back("yes");
  CHECK(!tdata1.empty()); // Check if new data modified the tdata size

  GML::TDATA tdata2("Apple", data);
  CHECK(!tdata2.empty());
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
  REQUIRE(!nodedata1.empty());
  REQUIRE(!tree1.empty());

  SUBCASE("Test wether tree was correctly move constructed") {
    auto tree2(std::move(tree1));
    auto nodedata2 = tree2.predict(data).nodedata();
    CHECK(!nodedata2.empty()); // Does the moved tree still works?
    CHECK(tree1.empty()); // Does the previous location still stores data?
  }


  SUBCASE("Test wether tree was correctly copy constructed") {
    auto tree2(tree1);
    auto nodedata2 = tree2.predict(data).nodedata();
    CHECK(!nodedata2.empty()); // Does the moved tree still works?
    CHECK(!tree1.empty()); // Does the previous location still stores data?
  }

  SUBCASE("Test tree copy assignment operation") {
    GML::TREE<std::string> tree2;
    tree2 = tree1;
    CHECK(!tree1.empty());
  }

  SUBCASE("Test tree move assignment operation") {
    GML::TREE<std::string> tree2;
    tree2 = std::move(tree1);
    CHECK(tree1.empty());
  }
}
