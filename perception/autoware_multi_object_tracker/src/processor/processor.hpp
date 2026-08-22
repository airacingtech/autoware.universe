// Copyright 2024 TIER IV, Inc.
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

#ifndef PROCESSOR__PROCESSOR_HPP_
#define PROCESSOR__PROCESSOR_HPP_

#include "autoware/multi_object_tracker/association/adaptive_threshold_cache.hpp"
#include "autoware/multi_object_tracker/association/association_manager.hpp"
#include "autoware/multi_object_tracker/association/tracker_overlap_manager.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/tracker_base.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <autoware_utils_debug/time_keeper.hpp>
#include <rclcpp/rclcpp.hpp>

#include "autoware_perception_msgs/msg/detected_objects.hpp"
#include "autoware_perception_msgs/msg/tracked_objects.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::multi_object_tracker
{
class TrackerProcessor
{
public:
  TrackerProcessor(
    const TrackerConfigs & tracker_configs, const TrackerCreationConfig & creation_config,
    const TrackerAssociationConfig & association_config,
    const TrackerOverlapManagerConfig & tracker_overlap_manager_config,
    const std::vector<types::InputChannel> & channels_config, const rclcpp::Logger & logger,
    rclcpp::Clock::SharedPtr clock);

  const std::list<std::shared_ptr<Tracker>> & getListTracker() const { return list_tracker_; }

  // Tracker processes
  void updateEgoPose(const std::optional<geometry_msgs::msg::PoseStamped> & ego_pose_stamped);
  void predictTrackers(const rclcpp::Time & time);
  types::AssociationResult associate(const types::DynamicObjectList & detected_objects) const;
  void update(const types::AssociatedObjects & associated_objects);
  void spawn(const types::AssociatedObjects & associated_objects);
  void prune(const rclcpp::Time & time);

  // Output processes
  void getTrackedObjects(
    const rclcpp::Time & time,
    autoware_perception_msgs::msg::TrackedObjects & tracked_objects) const;
  void getTentativeObjects(
    const rclcpp::Time & time,
    autoware_perception_msgs::msg::TrackedObjects & tentative_objects) const;
  void getMergedObjects(
    const rclcpp::Time & time, const geometry_msgs::msg::Transform & tf_base_to_world,
    autoware_perception_msgs::msg::DetectedObjects & merged_objects) const;

  void setTimeKeeper(std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr);

private:
  struct BirthHypothesis
  {
    uint channel_index;
    classes::Label label;
    geometry_msgs::msg::Point position;
    rclcpp::Time last_observation_time;
    int confirmation_count;
  };

  const TrackerConfigs tracker_configs_;
  const TrackerCreationConfig creation_config_;
  const std::vector<types::InputChannel> & channels_config_;

  std::optional<geometry_msgs::msg::PoseStamped> ego_pose_;

  std::unique_ptr<AssociationManager> association_manager_;
  std::unique_ptr<TrackerOverlapManager> tracker_overlap_manager_;

  mutable rclcpp::Time last_prune_time_;

  std::list<std::shared_ptr<Tracker>> list_tracker_;
  std::list<BirthHypothesis> birth_hypotheses_;
  uint64_t birth_guard_quarantined_count_{0};
  uint64_t birth_guard_released_count_{0};
  uint64_t birth_guard_expired_count_{0};
  uint64_t birth_guard_no_ego_withheld_count_{0};
  uint64_t birth_guard_nonfinite_rejected_count_{0};
  std::optional<geometry_msgs::msg::Pose> getEgoPose() const;
  void removeOldTracker(const rclcpp::Time & time);
  std::shared_ptr<Tracker> createNewTracker(
    const types::DynamicObject & object, const rclcpp::Time & time) const;
  void addTracker(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel & channel_config);
  void pruneBirthHypotheses(
    const rclcpp::Time & time, uint channel_index,
    const types::InputChannel::BirthGuard & config);
  std::list<BirthHypothesis>::iterator findBirthHypothesis(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel::BirthGuard & config);
  bool conflictsWithCoastingTracker(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel::BirthGuard & config) const;
  void logBirthGuardStats(const char * event, uint channel_index) const;

  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_;
  AdaptiveThresholdCache adaptive_threshold_cache_;

  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;
};

}  // namespace autoware::multi_object_tracker

#endif  // PROCESSOR__PROCESSOR_HPP_
