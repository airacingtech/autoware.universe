// Copyright 2026 Alex Nan
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

#include "../src/processor/input_manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

namespace autoware::multi_object_tracker
{
namespace
{

using namespace std::chrono_literals;

TEST(InputStream, ConsumesAnItemAtTheInclusiveCutoffExactlyOnce)
{
  types::InputChannel channel;
  channel.index = 0;
  channel.long_name = "test camera";
  const auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  InputStream stream(channel, nullptr, rclcpp::get_logger("input_manager_test"), clock);

  const rclcpp::Time stamp = clock->now();
  types::DynamicObjectList objects;
  objects.header.stamp = stamp;
  objects.channel_index = channel.index;
  stream.push(objects, types::AssociationResult{});

  types::ObjectsWithAssociationList first_export;
  stream.getObjectsOlderThan(stamp, stamp - rclcpp::Duration(1s), first_export);
  ASSERT_EQ(first_export.size(), 1U);

  types::ObjectsWithAssociationList second_export;
  stream.getObjectsOlderThan(stamp, stamp - rclcpp::Duration(1s), second_export);
  EXPECT_TRUE(second_export.empty());
}

}  // namespace
}  // namespace autoware::multi_object_tracker
