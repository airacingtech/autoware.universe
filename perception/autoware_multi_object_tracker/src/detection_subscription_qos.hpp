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

#ifndef DETECTION_SUBSCRIPTION_QOS_HPP_
#define DETECTION_SUBSCRIPTION_QOS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <cstddef>
#include <stdexcept>

namespace autoware::multi_object_tracker::input_qos
{

constexpr int kDepthMin = 1;
constexpr int kDepthMax = 1000;
constexpr int kDepthDefault = 1;

inline bool isValidDepth(const int requested)
{
  return requested >= kDepthMin && requested <= kDepthMax;
}

inline int requireValidDepth(const int requested)
{
  if (!isValidDepth(requested)) {
    throw std::invalid_argument("detection subscription depth is outside the supported range");
  }
  return requested;
}

inline rclcpp::QoS detectionQos(const int depth)
{
  return rclcpp::QoS{static_cast<size_t>(requireValidDepth(depth))};
}

}  // namespace autoware::multi_object_tracker::input_qos

#endif  // DETECTION_SUBSCRIPTION_QOS_HPP_
