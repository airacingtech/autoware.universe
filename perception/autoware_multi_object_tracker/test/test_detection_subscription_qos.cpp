// Copyright 2020 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../src/detection_subscription_qos.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace input_qos = ::autoware::multi_object_tracker::input_qos;

TEST(DetectionSubscriptionQos, PreservesTheUpstreamDefault)
{
  EXPECT_EQ(input_qos::kDepthDefault, 1);
  EXPECT_EQ(input_qos::detectionQos(input_qos::kDepthDefault).depth(), 1u);
}

TEST(DetectionSubscriptionQos, AcceptsUsefulKeepLastDepths)
{
  for (const int depth : {1, 2, 10, 100, 1000}) {
    EXPECT_TRUE(input_qos::isValidDepth(depth));
    const auto qos = input_qos::detectionQos(depth);
    EXPECT_EQ(qos.depth(), static_cast<size_t>(depth));
    EXPECT_EQ(qos.history(), rclcpp::HistoryPolicy::KeepLast);
  }
}

TEST(DetectionSubscriptionQos, RefusesInvalidDepths)
{
  for (const int depth : {
         0, -1, 1001, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
    EXPECT_FALSE(input_qos::isValidDepth(depth));
    EXPECT_THROW(input_qos::requireValidDepth(depth), std::invalid_argument);
    EXPECT_THROW(input_qos::detectionQos(depth), std::invalid_argument);
  }
}
