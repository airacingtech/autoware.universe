// Parameter validation and QoS construction for the delivery experiment.
#include <gtest/gtest.h>

#include "../src/experiment_qos.hpp"

#include <limits>

// `exp` would collide with std::exp under the test TU's using-directives.
namespace xq = ::autoware::multi_object_tracker::experiment;

TEST(ExperimentQos, TheDefaultIsTheShippedDepth)
{
  EXPECT_EQ(xq::kSubDepthDefault, 1) << "an unconfigured node must behave as upstream";
  EXPECT_EQ(xq::validated_sub_depth(xq::kSubDepthDefault), 1);
  EXPECT_EQ(xq::detection_qos(xq::kSubDepthDefault).depth(), 1u);
}

TEST(ExperimentQos, AcceptsTheWholeValidRange)
{
  for (int d : {1, 2, 10, 100, 999, 1000}) {
    EXPECT_TRUE(xq::is_valid_sub_depth(d)) << d;
    EXPECT_EQ(xq::validated_sub_depth(d), d) << d;
    EXPECT_EQ(xq::detection_qos(d).depth(), static_cast<size_t>(d)) << d;
  }
}

TEST(ExperimentQos, RefusesZeroNegativeAndOutOfRange)
{
  for (int d : {0, -1, -1000, 1001, 100000}) {
    EXPECT_FALSE(xq::is_valid_sub_depth(d)) << d;
    EXPECT_EQ(xq::validated_sub_depth(d), xq::kSubDepthDefault) << d;
    // ...and the QoS built from a refused value is the shipped one, never depth 0
    EXPECT_EQ(xq::detection_qos(d).depth(), 1u) << d;
  }
}

TEST(ExperimentQos, RefusesIntegerExtremes)
{
  EXPECT_FALSE(xq::is_valid_sub_depth(std::numeric_limits<int>::max()));
  EXPECT_FALSE(xq::is_valid_sub_depth(std::numeric_limits<int>::min()));
  EXPECT_EQ(xq::validated_sub_depth(std::numeric_limits<int>::max()), 1);
  EXPECT_EQ(xq::validated_sub_depth(std::numeric_limits<int>::min()), 1);
}

TEST(ExperimentQos, TheQosIsKeepLastAtTheRequestedDepth)
{
  const auto q = xq::detection_qos(10);
  EXPECT_EQ(q.depth(), 10u);
  EXPECT_EQ(q.history(), rclcpp::HistoryPolicy::KeepLast);
}
