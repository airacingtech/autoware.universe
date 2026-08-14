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

// EXPERIMENT SUPPORT: detection-subscription history depth.
//
// Separated from the node so the validation and the QoS construction can be unit
// tested without standing up a node. Nothing here changes shipped behaviour: the
// default is the shipped depth of 1.
#ifndef EXPERIMENT_QOS_HPP_
#define EXPERIMENT_QOS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <cstddef>

namespace autoware::multi_object_tracker::experiment
{

constexpr int kSubDepthMin = 1;
constexpr int kSubDepthMax = 1000;
constexpr int kSubDepthDefault = 1;

// TRUE when the requested depth is a usable KEEP_LAST history depth. Integer only
// (the caller's parameter type enforces that), and bounded: zero, negative and
// absurdly large values are all refused rather than coerced, because a silently
// coerced depth would make an A/B report a depth it never ran at.
inline bool is_valid_sub_depth(int requested)
{
  return requested >= kSubDepthMin && requested <= kSubDepthMax;
}

// The validated depth, falling back to the SHIPPED default when the request is
// unusable. Never returns anything outside [kSubDepthMin, kSubDepthMax].
inline int validated_sub_depth(int requested)
{
  return is_valid_sub_depth(requested) ? requested : kSubDepthDefault;
}

// The detection subscription's QoS for a given depth. KEEP_LAST with an explicit
// depth; reliability is left at the profile default so this experiment changes
// one variable and one only.
inline rclcpp::QoS detection_qos(int depth)
{
  return rclcpp::QoS{static_cast<size_t>(validated_sub_depth(depth))};
}

}  // namespace autoware::multi_object_tracker::experiment

#endif  // EXPERIMENT_QOS_HPP_
