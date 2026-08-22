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

#include "processor.hpp"

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/tracker/tracker.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/static_tracker.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <tf2/transform_datatypes.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
using autoware_utils_debug::ScopedTimeTrack;

TrackerProcessor::TrackerProcessor(
  const TrackerConfigs & tracker_configs, const TrackerCreationConfig & creation_config,
  const TrackerAssociationConfig & association_config,
  const TrackerOverlapManagerConfig & tracker_overlap_manager_config,
  const std::vector<types::InputChannel> & channels_config, const rclcpp::Logger & logger,
  rclcpp::Clock::SharedPtr clock)
: tracker_configs_(tracker_configs),
  creation_config_(creation_config),
  channels_config_(channels_config),
  logger_(logger),
  clock_(std::move(clock))
{
  association_manager_ = std::make_unique<AssociationManager>(association_config, channels_config);
  tracker_overlap_manager_ =
    std::make_unique<TrackerOverlapManager>(tracker_overlap_manager_config);
}

std::optional<geometry_msgs::msg::Pose> TrackerProcessor::getEgoPose() const
{
  return ego_pose_ ? std::make_optional(ego_pose_->pose) : std::nullopt;
}

void TrackerProcessor::updateEgoPose(
  const std::optional<geometry_msgs::msg::PoseStamped> & ego_pose_stamped)
{
  ego_pose_ = ego_pose_stamped;
}

void TrackerProcessor::predictTrackers(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  for (auto itr = list_tracker_.begin(); itr != list_tracker_.end(); ++itr) {
    (*itr)->predict(time);
  }
}

types::AssociationResult TrackerProcessor::associate(
  const types::DynamicObjectList & detected_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  return association_manager_->associate(detected_objects, list_tracker_, ego_pose_);
}

void TrackerProcessor::update(const types::AssociatedObjects & associated_objects)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  const auto & detected_objects = associated_objects.objects;
  const auto & association_result = associated_objects.association;

  int tracker_idx = 0;
  const auto & time = detected_objects.header.stamp;
  for (auto tracker_itr = list_tracker_.begin(); tracker_itr != list_tracker_.end();
       ++tracker_itr, ++tracker_idx) {
    bool found = false;
    size_t measurement_idx = 0;
    unique_identifier_msgs::msg::UUID tracker_uuid = (*tracker_itr)->getUUID();

    if (association_result.tracker_to_measurement.count(tracker_uuid)) {
      unique_identifier_msgs::msg::UUID measurement_uuid =
        association_result.tracker_to_measurement.at(tracker_uuid);
      const auto idx = detected_objects.getObjectIndexByUuid(measurement_uuid);
      if (idx) {
        measurement_idx = *idx;
        found = true;
      }
    }

    if (found) {
      const auto & associated_object = detected_objects.objects.at(measurement_idx);
      const types::InputChannel channel_info = channels_config_[associated_object.channel_index];
      const auto update_guard_result = evaluateAssociatedUpdate(
        *tracker_itr, associated_object, time, channel_info.birth_guard.update_guard);
      if (update_guard_result.reject) {
        // Keep the association intact: spawn() will see measurement_to_tracker and cannot create a
        // second UUID from the rejected alternate depth mode in this cycle.  Only the normal
        // bounded no-measurement path is applied to the established tracker.
        (*tracker_itr)->updateWithoutMeasurement(time);
        ++update_guard_rejected_count_;
        if (update_guard_result.used_no_ego_fallback) {
          ++update_guard_no_ego_rejected_count_;
        }
        if (update_guard_result.invalid_input) {
          ++update_guard_invalid_rejected_count_;
        }
        logBirthGuardStats("associated_update_rejected", associated_object.channel_index);
        RCLCPP_WARN_THROTTLE(
          logger_, *clock_, 1000,
          "Withheld associated camera update on channel %u for tracker %s: radial/XY "
          "innovation=%.2f m allowance=%.2f m bearing_delta=%.2f deg; tracker is coasting and "
          "the consumed measurement cannot spawn a new UUID in this cycle",
          associated_object.channel_index, (*tracker_itr)->getUuidString().c_str(),
          update_guard_result.innovation_m, update_guard_result.allowance_m,
          update_guard_result.bearing_difference_deg);
        continue;
      }
      const bool has_significant_shape_change = association_result.wasShapeChanged(tracker_uuid);
      (*tracker_itr)
        ->setEgoPose(ego_pose_ ? std::make_optional(ego_pose_->pose.position) : std::nullopt);
      (*(tracker_itr))
        ->updateWithMeasurement(
          associated_object, time, channel_info, has_significant_shape_change);
    } else {
      (*(tracker_itr))->updateWithoutMeasurement(time);
    }
  }
}

TrackerProcessor::UpdateGuardResult TrackerProcessor::evaluateAssociatedUpdate(
  const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & measurement,
  const rclcpp::Time & time, const types::InputChannel::BirthGuard::UpdateGuard & config) const
{
  UpdateGuardResult result;
  if (!config.enabled) {
    return result;
  }

  types::DynamicObject prediction;
  if (!tracker->getTrackedObject(time, prediction, false)) {
    result.reject = true;
    result.invalid_input = true;
    return result;
  }

  const auto finite_xy = [](const geometry_msgs::msg::Point & point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
  };
  if (!finite_xy(measurement.pose.position) || !finite_xy(prediction.pose.position)) {
    result.reject = true;
    result.invalid_input = true;
    return result;
  }

  const double elapsed_sec = tracker->getElapsedTimeFromLastUpdate(time);
  if (elapsed_sec < 0.0 || !std::isfinite(elapsed_sec)) {
    result.reject = true;
    result.invalid_input = true;
    return result;
  }
  const double bounded_elapsed_sec = std::min(elapsed_sec, config.max_elapsed_sec);
  result.allowance_m = std::min(
    config.max_allowance_m,
    config.base_allowance_m + config.max_innovation_speed_mps * bounded_elapsed_sec);
  if (!std::isfinite(result.allowance_m)) {
    result.reject = true;
    result.invalid_input = true;
    return result;
  }

  const double dx = measurement.pose.position.x - prediction.pose.position.x;
  const double dy = measurement.pose.position.y - prediction.pose.position.y;
  const double euclidean_innovation = std::hypot(dx, dy);
  if (!std::isfinite(euclidean_innovation)) {
    result.reject = true;
    result.invalid_input = true;
    return result;
  }
  const bool has_usable_ego_pose = ego_pose_ && finite_xy(ego_pose_->pose.position);

  // Velocity needs multiple accepted measurements to bootstrap.  Before that point, do not apply
  // the tighter velocity-relative radial gate, but still reject a mode switch beyond a loose hard
  // displacement cap.  The same cap closes the discontinuity just outside max_bearing_deg.
  if (euclidean_innovation > config.max_euclidean_innovation_m) {
    result.used_no_ego_fallback = !has_usable_ego_pose;
    result.innovation_m = euclidean_innovation;
    result.allowance_m = config.max_euclidean_innovation_m;
    result.reject = true;
    return result;
  }
  if (tracker->getTotalMeasurementCount() < config.min_measurements) {
    return result;
  }

  if (!has_usable_ego_pose) {
    // Fail closed on a physically impossible associated displacement when odometry/TF is absent.
    // The fallback is intentionally the looser Euclidean cap because bearing/range cannot be
    // established safely; the tighter radial allowance has no valid geometry here.
    result.used_no_ego_fallback = true;
    result.innovation_m = euclidean_innovation;
    result.allowance_m = config.max_euclidean_innovation_m;
    result.reject = result.innovation_m > config.max_euclidean_innovation_m;
    return result;
  }

  const auto & ego = ego_pose_->pose.position;
  const double tracker_x = prediction.pose.position.x - ego.x;
  const double tracker_y = prediction.pose.position.y - ego.y;
  const double measurement_x = measurement.pose.position.x - ego.x;
  const double measurement_y = measurement.pose.position.y - ego.y;
  const double tracker_range = std::hypot(tracker_x, tracker_y);
  const double measurement_range = std::hypot(measurement_x, measurement_y);
  if (
    !std::isfinite(tracker_range) || !std::isfinite(measurement_range) || tracker_range <= 1e-6 ||
    measurement_range <= 1e-6) {
    // Degenerate ego-relative geometry still gets the bounded Euclidean check instead of silently
    // failing open.
    result.used_no_ego_fallback = true;
    result.innovation_m = euclidean_innovation;
    result.allowance_m = config.max_euclidean_innovation_m;
    result.reject = result.innovation_m > config.max_euclidean_innovation_m;
    return result;
  }

  const double cross = tracker_x * measurement_y - tracker_y * measurement_x;
  const double dot = tracker_x * measurement_x + tracker_y * measurement_y;
  const double bearing_difference = std::abs(std::atan2(cross, dot));
  constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;
  result.bearing_difference_deg = bearing_difference * radians_to_degrees;
  if (!std::isfinite(result.bearing_difference_deg)) {
    result.reject = true;
    result.invalid_input = true;
    return result;
  }
  if (result.bearing_difference_deg > config.max_bearing_deg) {
    return result;
  }

  // The prediction already contains the tracker's estimated velocity and process covariance.
  // Association remains the statistical covariance gate.  This extra camera-only check is a
  // physical bound on unexplained radial motion; allowing raw monocular depth covariance here
  // would make alternate ray/mesh modes fail open again.
  result.innovation_m = std::abs(measurement_range - tracker_range);
  result.reject = result.innovation_m > result.allowance_m;
  return result;
}

void TrackerProcessor::spawn(const types::AssociatedObjects & associated_objects)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  const auto & detected_objects = associated_objects.objects;
  const auto & association_result = associated_objects.association;

  const auto channel_config = channels_config_[detected_objects.channel_index];
  if (!channel_config.is_spawn_enabled) {
    return;
  }

  const auto & time = detected_objects.header.stamp;
  if (channel_config.birth_guard.enabled) {
    pruneBirthHypotheses(time, detected_objects.channel_index, channel_config.birth_guard);
  }

  for (size_t i = 0; i < detected_objects.objects.size(); ++i) {
    const auto & new_object = detected_objects.objects.at(i);
    if (association_result.measurement_to_tracker.count(new_object.uuid)) {
      continue;
    }

    if (!channel_config.birth_guard.enabled) {
      addTracker(new_object, time, channel_config);
      continue;
    }

    if (!std::isfinite(new_object.pose.position.x) ||
        !std::isfinite(new_object.pose.position.y)) {
      ++birth_guard_nonfinite_rejected_count_;
      logBirthGuardStats("rejected_nonfinite_measurement", new_object.channel_index);
      continue;
    }

    const bool has_usable_ego_pose =
      ego_pose_ && std::isfinite(ego_pose_->pose.position.x) &&
      std::isfinite(ego_pose_->pose.position.y);
    if (!has_usable_ego_pose) {
      // Association already decided that this measurement cannot update an existing tracker.  If
      // ego pose is missing, a range/bearing conflict cannot be ruled out, so fail closed only for
      // tracker birth. Existing trackers still follow their normal bounded prediction/coast path.
      ++birth_guard_no_ego_withheld_count_;
      logBirthGuardStats("withheld_no_ego", new_object.channel_index);
    }
    const bool has_conflict =
      !has_usable_ego_pose ||
      conflictsWithCoastingTracker(new_object, time, channel_config.birth_guard);
    auto hypothesis = findBirthHypothesis(new_object, time, channel_config.birth_guard);

    if (hypothesis == birth_hypotheses_.end()) {
      if (!has_conflict) {
        // The guard is intentionally not a global camera birth delay.  A non-conflicting object
        // keeps the normal spawn behavior, including a genuine second car seen beside a tracker
        // that was successfully updated in this frame.
        addTracker(new_object, time, channel_config);
        continue;
      }

      birth_hypotheses_.push_back(BirthHypothesis{
        new_object.channel_index, classes::getHighestProbLabel(new_object.classification),
        new_object.pose.position, time, 1});
      ++birth_guard_quarantined_count_;
      logBirthGuardStats("quarantined", new_object.channel_index);
      RCLCPP_DEBUG(
        logger_,
        "Quarantined unmatched %s birth at (%.2f, %.2f): farther same-bearing measurement "
        "conflicts with a coasting established tracker",
        classes::toString(classes::getHighestProbLabel(new_object.classification)).c_str(),
        new_object.pose.position.x, new_object.pose.position.y);
      continue;
    }

    hypothesis->position = new_object.pose.position;
    hypothesis->last_observation_time = time;
    ++hypothesis->confirmation_count;

    // Confirmation alone must never override an active depth conflict.  It only permits birth
    // once the old tracker was observed again (so this can be a real second object) or its bounded
    // coast ended and the tracker was pruned.
    if (
      has_conflict ||
      hypothesis->confirmation_count < channel_config.birth_guard.min_confirmations)
    {
      continue;
    }

    birth_hypotheses_.erase(hypothesis);
    ++birth_guard_released_count_;
    logBirthGuardStats("released", new_object.channel_index);
    addTracker(new_object, time, channel_config);
  }
}

void TrackerProcessor::addTracker(
  const types::DynamicObject & object, const rclcpp::Time & time,
  const types::InputChannel & channel_config)
{
  std::shared_ptr<Tracker> tracker = createNewTracker(object, time);
  if (!tracker) return;  // null combo: (shape, label) not accepted

  const float initial_existence_probability = channel_config.trust_existence_probability
                                                ? object.existence_probability
                                                : types::default_existence_probability;
  tracker->initializeExistenceProbabilities(
    object.channel_index, initial_existence_probability);
  list_tracker_.push_back(tracker);
}

void TrackerProcessor::pruneBirthHypotheses(
  const rclcpp::Time & time, const uint channel_index,
  const types::InputChannel::BirthGuard & config)
{
  const size_t size_before = birth_hypotheses_.size();
  birth_hypotheses_.remove_if([&](const BirthHypothesis & hypothesis) {
    if (hypothesis.channel_index != channel_index) return false;
    const double age = (time - hypothesis.last_observation_time).seconds();
    return age < 0.0 || age > config.hypothesis_timeout_sec;
  });
  const size_t expired_count = size_before - birth_hypotheses_.size();
  if (expired_count > 0) {
    birth_guard_expired_count_ += expired_count;
    logBirthGuardStats("expired", channel_index);
  }
}

std::list<TrackerProcessor::BirthHypothesis>::iterator TrackerProcessor::findBirthHypothesis(
  const types::DynamicObject & object, const rclcpp::Time & time,
  const types::InputChannel::BirthGuard & config)
{
  const auto label = classes::getHighestProbLabel(object.classification);
  auto nearest = birth_hypotheses_.end();
  double nearest_distance_sq = std::numeric_limits<double>::infinity();

  for (auto it = birth_hypotheses_.begin(); it != birth_hypotheses_.end(); ++it) {
    if (it->channel_index != object.channel_index || it->label != label) continue;

    const double dt = (time - it->last_observation_time).seconds();
    // dt == 0 also prevents two objects in one message from consuming one hypothesis.
    if (dt <= 0.0 || dt > config.hypothesis_timeout_sec) continue;

    const double dx = object.pose.position.x - it->position.x;
    const double dy = object.pose.position.y - it->position.y;
    const double distance_sq = dx * dx + dy * dy;
    const double max_distance =
      config.hypothesis_match_distance_m + config.hypothesis_max_speed_mps * dt;
    if (distance_sq <= max_distance * max_distance && distance_sq < nearest_distance_sq) {
      nearest = it;
      nearest_distance_sq = distance_sq;
    }
  }
  return nearest;
}

bool TrackerProcessor::conflictsWithCoastingTracker(
  const types::DynamicObject & object, const rclcpp::Time & time,
  const types::InputChannel::BirthGuard & config) const
{
  if (
    !ego_pose_ || !std::isfinite(ego_pose_->pose.position.x) ||
    !std::isfinite(ego_pose_->pose.position.y)) {
    return true;
  }

  const auto measurement_label = classes::getHighestProbLabel(object.classification);
  const auto & ego = ego_pose_->pose.position;
  const double measurement_x = object.pose.position.x - ego.x;
  const double measurement_y = object.pose.position.y - ego.y;
  const double measurement_range = std::hypot(measurement_x, measurement_y);
  if (measurement_range <= config.conflict_min_range_gap_m) return false;

  constexpr double degrees_to_radians = 3.14159265358979323846 / 180.0;
  const double max_bearing = config.conflict_max_bearing_deg * degrees_to_radians;

  for (const auto & tracker : list_tracker_) {
    if (
      tracker->getTotalMeasurementCount() < config.min_established_measurements ||
      tracker->getNoMeasurementCount() == 0 ||
      tracker->getHighestProbLabel() != measurement_label)
    {
      continue;
    }

    const double coast_age = tracker->getElapsedTimeFromLastUpdate(time);
    if (coast_age < 0.0 || coast_age > config.conflict_max_coast_age_sec) continue;
    if (!tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), time)) continue;

    types::DynamicObject prediction;
    if (!tracker->getTrackedObject(time, prediction, false)) continue;

    const double tracker_x = prediction.pose.position.x - ego.x;
    const double tracker_y = prediction.pose.position.y - ego.y;
    const double tracker_range = std::hypot(tracker_x, tracker_y);
    if (tracker_range <= 1e-6) continue;

    const double range_gap = measurement_range - tracker_range;
    if (range_gap < config.conflict_min_range_gap_m) continue;

    const double cross = tracker_x * measurement_y - tracker_y * measurement_x;
    const double dot = tracker_x * measurement_x + tracker_y * measurement_y;
    const double bearing_difference = std::abs(std::atan2(cross, dot));
    if (bearing_difference <= max_bearing) return true;
  }
  return false;
}

void TrackerProcessor::logBirthGuardStats(const char * event, const uint channel_index) const
{
  RCLCPP_INFO_THROTTLE(
    logger_, *clock_, 1000,
    "Birth guard %s on channel %u: quarantined=%llu released=%llu expired=%llu "
    "no_ego_withheld=%llu nonfinite_rejected=%llu active=%zu update_rejected=%llu "
    "update_no_ego_rejected=%llu update_invalid_rejected=%llu",
    event, channel_index, static_cast<unsigned long long>(birth_guard_quarantined_count_),
    static_cast<unsigned long long>(birth_guard_released_count_),
    static_cast<unsigned long long>(birth_guard_expired_count_),
    static_cast<unsigned long long>(birth_guard_no_ego_withheld_count_),
    static_cast<unsigned long long>(birth_guard_nonfinite_rejected_count_),
    birth_hypotheses_.size(), static_cast<unsigned long long>(update_guard_rejected_count_),
    static_cast<unsigned long long>(update_guard_no_ego_rejected_count_),
    static_cast<unsigned long long>(update_guard_invalid_rejected_count_));
}

std::shared_ptr<Tracker> TrackerProcessor::createNewTracker(
  const types::DynamicObject & object, const rclcpp::Time & time) const
{
  const classes::Label label = classes::getHighestProbLabel(object.classification);
  const ShapeLabelKey key{types::toShapeType(object.shape.type), label};

  const auto tracker_type_opt = get_map_value_if_exists(creation_config_.shape_tracker_map, key);

  if (tracker_type_opt) {
    switch (tracker_type_opt->get()) {
      case types::TrackerType::MULTIPLE_VEHICLE:
        return std::make_shared<MultipleVehicleTracker>(time, object);
      case types::TrackerType::GENERAL_VEHICLE:
        return std::make_shared<VehicleTracker>(object_model::general_vehicle, time, object);
      case types::TrackerType::PEDESTRIAN_AND_BICYCLE:
        return std::make_shared<PedestrianAndBicycleTracker>(time, object);
      case types::TrackerType::NORMAL_VEHICLE:
        return std::make_shared<VehicleTracker>(object_model::normal_vehicle, time, object);
      case types::TrackerType::PEDESTRIAN:
        return std::make_shared<PedestrianTracker>(time, object);
      case types::TrackerType::BICYCLE:
        return std::make_shared<VehicleTracker>(object_model::bicycle, time, object);
      case types::TrackerType::BIG_VEHICLE:
        return std::make_shared<VehicleTracker>(object_model::big_vehicle, time, object);
      case types::TrackerType::STATIC:
        return std::make_shared<StaticTracker>(time, object, tracker_configs_.static_tracker);
      case types::TrackerType::POLYGON:
        return std::make_shared<PolygonTracker>(time, object, tracker_configs_.polygon_tracker);
      default:
        return std::make_shared<PolygonTracker>(time, object, tracker_configs_.polygon_tracker);
    }
  }

  if (creation_config_.explicit_null_combos.count(key)) {
    return nullptr;  // create: "null" — explicitly not accepted, silently skip
  }

  // implicitly omitted — not listed in tracker_assignment; error periodically
  RCLCPP_ERROR_THROTTLE(
    logger_, *clock_, 1000,
    "Received detection with unspecified tracker_assignment combination: shape=%s label=%s. "
    "Add an explicit entry (or create: \"null\") to suppress this error.",
    types::toString(key.first).c_str(), classes::toString(key.second).c_str());
  return nullptr;
}

void TrackerProcessor::prune(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  if (time.nanoseconds() - last_prune_time_.nanoseconds() < 2000 /*2ms*/) {
    return;
  }

  removeOldTracker(time);
  tracker_overlap_manager_->merge(list_tracker_, time, adaptive_threshold_cache_, getEgoPose());

  last_prune_time_ = time;
}

void TrackerProcessor::removeOldTracker(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  for (auto itr = list_tracker_.begin(); itr != list_tracker_.end(); ++itr) {
    if ((*itr)->isExpired(time, adaptive_threshold_cache_, getEgoPose())) {
      auto erase_itr = itr;
      --itr;
      list_tracker_.erase(erase_itr);
    }
  }
}

void TrackerProcessor::getTrackedObjects(
  const rclcpp::Time & time, autoware_perception_msgs::msg::TrackedObjects & tracked_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  tracked_objects.header.stamp = time;
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    if (!tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt)) continue;
    constexpr bool to_publish = true;
    if (tracker->getTrackedObject(time, tracked_object, to_publish)) {
      tracked_object.existence_probability = tracker->getTotalExistenceProbability();
      tracked_object.classification = tracker->getClassification();
      tracked_objects.objects.push_back(types::toTrackedObjectMsg(tracked_object));
    }
  }
}

void TrackerProcessor::getTentativeObjects(
  const rclcpp::Time & time,
  autoware_perception_msgs::msg::TrackedObjects & tentative_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  tentative_objects.header.stamp = time;
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    if (tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt)) continue;
    constexpr bool to_publish = false;
    if (tracker->getTrackedObject(time, tracked_object, to_publish)) {
      tentative_objects.objects.push_back(types::toTrackedObjectMsg(tracked_object));
    }
  }
}

void TrackerProcessor::getMergedObjects(
  const rclcpp::Time & time, const geometry_msgs::msg::Transform & tf_base_to_world,
  autoware_perception_msgs::msg::DetectedObjects & merged_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  merged_objects.header.stamp = time;
  merged_objects.objects.clear();
  merged_objects.objects.reserve(list_tracker_.size());
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    constexpr bool to_publish = false;
    if (tracker->getTrackedObject(time, tracked_object, to_publish)) {
      merged_objects.objects.push_back(types::toDetectedObjectMsg(tracked_object));
    }
  }

  // Transform poses from world frame to ego frame using the inverse of tf_base_to_world
  tf2::Transform tf2_base_to_world;
  tf2::fromMsg(tf_base_to_world, tf2_base_to_world);
  geometry_msgs::msg::TransformStamped ts;
  ts.transform = tf2::toMsg(tf2_base_to_world.inverse());

  for (auto & obj : merged_objects.objects) {
    tf2::doTransform(
      obj.kinematics.pose_with_covariance.pose, obj.kinematics.pose_with_covariance.pose, ts);
  }
}

void TrackerProcessor::setTimeKeeper(
  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr)
{
  time_keeper_ = std::move(time_keeper_ptr);
  association_manager_->setTimeKeeper(time_keeper_);
}

}  // namespace autoware::multi_object_tracker
