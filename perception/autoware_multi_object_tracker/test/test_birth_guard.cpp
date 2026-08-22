// Copyright 2026 TIER IV, Inc.
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

#include "../src/processor/processor.hpp"
#include "test_bench.hpp"
#include "test_utils.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/shape.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace mot = autoware::multi_object_tracker;
using std::chrono_literals::operator""ms;

class BirthGuardTest : public ::testing::Test
{
protected:
  using Processor = mot::TrackerProcessor;

  void SetUp() override
  {
    tracker_configs_ = createTrackerConfigs();
    creation_config_ = createTrackerCreationConfig();
    association_config_ = createTrackerAssociationConfig();
    overlap_config_ = createTrackerOverlapManagerConfig();
    channels_ = createInputChannelsConfig();
    channels_.front().trust_position_as_center = true;
    channels_.front().birth_guard.enabled = true;
    channels_.front().birth_guard.min_confirmations = 3;
    channels_.front().birth_guard.min_established_measurements = 2;
    channels_.front().birth_guard.hypothesis_timeout_sec = 0.35;
    channels_.front().birth_guard.hypothesis_match_distance_m = 1.0;
    channels_.front().birth_guard.hypothesis_max_speed_mps = 100.0;
    channels_.front().birth_guard.conflict_max_coast_age_sec = 0.8;
    channels_.front().birth_guard.conflict_min_range_gap_m = 12.0;
    channels_.front().birth_guard.conflict_max_bearing_deg = 2.0;
    channels_.front().birth_guard.update_guard.enabled = true;
    channels_.front().birth_guard.update_guard.min_measurements = 3;
    channels_.front().birth_guard.update_guard.base_allowance_m = 0.75;
    channels_.front().birth_guard.update_guard.max_innovation_speed_mps = 35.0;
    channels_.front().birth_guard.update_guard.max_elapsed_sec = 0.15;
    channels_.front().birth_guard.update_guard.max_allowance_m = 4.5;
    channels_.front().birth_guard.update_guard.max_bearing_deg = 2.0;
    channels_.front().birth_guard.update_guard.max_euclidean_innovation_m = 5.0;
    resetProcessor();
  }

  void resetProcessor()
  {
    processor_ = std::make_unique<Processor>(
      tracker_configs_, creation_config_, association_config_, overlap_config_, channels_,
      rclcpp::get_logger("birth_guard_test"),
      std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME));
  }

  static rclcpp::Time baseTime()
  {
    return rclcpp::Time(1000000000LL, RCL_ROS_TIME);
  }

  mot::types::DynamicObject makeCar(
    const double x, const double y, const rclcpp::Time & time, const std::string & id) const
  {
    mot::types::DynamicObject object;
    object.uuid.uuid = stringToUUID(id);
    object.time = time;
    object.channel_index = 0;
    object.existence_probability = 0.95F;
    object.classification = {{mot::classes::Label::CAR, 1.0F}};
    object.pose.position.x = x;
    object.pose.position.y = y;
    object.pose.orientation.w = 1.0;
    object.pose_covariance.fill(0.0);
    object.twist_covariance.fill(0.0);
    object.kinematics.orientation_availability = mot::types::OrientationAvailability::AVAILABLE;
    object.kinematics.has_position_covariance = false;
    object.kinematics.has_twist = false;
    object.kinematics.has_twist_covariance = false;
    object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object.shape.dimensions.x = 4.8;
    object.shape.dimensions.y = 2.0;
    object.shape.dimensions.z = 1.5;
    object.area = object.shape.dimensions.x * object.shape.dimensions.y;
    return object;
  }

  mot::types::DynamicObjectList makeObjects(
    const rclcpp::Time & time, std::vector<std::pair<double, double>> positions)
  {
    mot::types::DynamicObjectList objects;
    objects.header.stamp = time;
    objects.header.frame_id = "map";
    objects.channel_index = 0;
    for (const auto & [x, y] : positions) {
      objects.objects.push_back(makeCar(x, y, time, "measurement_" + std::to_string(next_id_++)));
    }
    return objects;
  }

  void process(
    const rclcpp::Time & time, std::vector<std::pair<double, double>> positions,
    const bool ego_available = true)
  {
    if (ego_available) {
      geometry_msgs::msg::PoseStamped ego;
      ego.header.stamp = time;
      ego.header.frame_id = "map";
      ego.pose.orientation.w = 1.0;
      processor_->updateEgoPose(ego);
    } else {
      processor_->updateEgoPose(std::nullopt);
    }
    processor_->predictTrackers(time);

    auto objects = makeObjects(time, std::move(positions));
    const auto association = processor_->associate(objects);
    const mot::types::AssociatedObjects associated{objects, association};
    processor_->update(associated);
    processor_->prune(time);
    processor_->spawn(associated);
  }

  void establishNearTracker(rclcpp::Time & time)
  {
    for (int i = 0; i < 4; ++i) {
      process(time, {{10.0, 0.0}});
      time += rclcpp::Duration(50ms);
    }
    ASSERT_EQ(processor_->getListTracker().size(), 1U);
  }

  void forceAssociatedUpdate(
    const rclcpp::Time & time, const std::pair<double, double> position,
    mot::types::DynamicObject * prediction_before_update = nullptr, const bool ego_available = true)
  {
    ASSERT_EQ(processor_->getListTracker().size(), 1U);
    if (ego_available) {
      geometry_msgs::msg::PoseStamped ego;
      ego.header.stamp = time;
      ego.header.frame_id = "map";
      ego.pose.orientation.w = 1.0;
      processor_->updateEgoPose(ego);
    } else {
      processor_->updateEgoPose(std::nullopt);
    }
    processor_->predictTrackers(time);

    const auto tracker = processor_->getListTracker().front();
    if (prediction_before_update) {
      ASSERT_TRUE(tracker->getTrackedObject(time, *prediction_before_update, false));
    }
    auto objects = makeObjects(time, {position});
    mot::types::AssociationResult association;
    association.add(tracker->getUUID(), objects.objects.front().uuid);
    const mot::types::AssociatedObjects associated{objects, association};
    processor_->update(associated);
    processor_->prune(time);
    processor_->spawn(associated);
  }

  mot::TrackerConfigs tracker_configs_;
  mot::TrackerCreationConfig creation_config_;
  mot::TrackerAssociationConfig association_config_;
  mot::TrackerOverlapManagerConfig overlap_config_;
  std::vector<mot::types::InputChannel> channels_;
  std::unique_ptr<Processor> processor_;
  int next_id_{0};
};

TEST_F(BirthGuardTest, DisabledChannelKeepsImmediateSpawnBehavior)
{
  channels_.front().birth_guard.enabled = false;
  resetProcessor();
  auto time = baseTime();
  establishNearTracker(time);

  process(time, {{50.0, 0.0}});

  EXPECT_EQ(processor_->getListTracker().size(), 2U);
}

TEST_F(BirthGuardTest, QuarantinesFarSameBearingBirthWhileEstablishedTrackerCoasts)
{
  auto time = baseTime();
  establishNearTracker(time);

  for (int i = 0; i < 4; ++i) {
    process(time, {{50.0, 0.0}});
    time += rclcpp::Duration(50ms);
  }

  EXPECT_EQ(processor_->getListTracker().size(), 1U);
}

TEST_F(BirthGuardTest, DoesNotBlockASecondCarWhenEstablishedTrackerWasObserved)
{
  auto time = baseTime();
  establishNearTracker(time);

  process(time, {{10.0, 0.0}, {50.0, 0.0}});

  EXPECT_EQ(processor_->getListTracker().size(), 2U);
}

TEST_F(BirthGuardTest, ReleasesConfirmedHypothesisAfterCoastConflictClears)
{
  auto time = baseTime();
  establishNearTracker(time);

  for (int i = 0; i < 3; ++i) {
    process(time, {{50.0, 0.0}});
    time += rclcpp::Duration(50ms);
  }
  ASSERT_EQ(processor_->getListTracker().size(), 1U);

  process(time, {{10.0, 0.0}, {50.0, 0.0}});

  EXPECT_EQ(processor_->getListTracker().size(), 2U);
}

TEST_F(BirthGuardTest, DoesNotTreatDifferentBearingAsDepthConflict)
{
  auto time = baseTime();
  establishNearTracker(time);

  process(time, {{0.0, 50.0}});

  EXPECT_EQ(processor_->getListTracker().size(), 2U);
}

TEST_F(BirthGuardTest, CollinearSecondCarDelayEndsAtConfiguredCoastBound)
{
  channels_.front().birth_guard.conflict_max_coast_age_sec = 0.2;
  resetProcessor();
  auto time = baseTime();
  establishNearTracker(time);

  for (int i = 0; i < 4; ++i) {
    process(time, {{50.0, 0.0}});
    time += rclcpp::Duration(50ms);
  }
  ASSERT_EQ(processor_->getListTracker().size(), 1U);

  process(time, {{50.0, 0.0}});

  EXPECT_EQ(processor_->getListTracker().size(), 2U);
}

TEST_F(BirthGuardTest, WithholdsImpossibleSameBearingAssociatedUpdateWithoutSpawning)
{
  auto time = baseTime();
  establishNearTracker(time);
  const auto tracker = processor_->getListTracker().front();
  const int measurements_before = tracker->getTotalMeasurementCount();

  mot::types::DynamicObject prediction_before_update;
  forceAssociatedUpdate(time, {14.5, 0.0}, &prediction_before_update);

  ASSERT_EQ(processor_->getListTracker().size(), 1U);
  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 1);
  mot::types::DynamicObject state_after_update;
  ASSERT_TRUE(tracker->getTrackedObject(time, state_after_update, false));
  EXPECT_NEAR(state_after_update.pose.position.x, prediction_before_update.pose.position.x, 1e-9);
  EXPECT_NEAR(state_after_update.pose.position.y, prediction_before_update.pose.position.y, 1e-9);
}

TEST_F(BirthGuardTest, AcceptsLaterValidUpdateAfterRejectedDepthMode)
{
  auto time = baseTime();
  establishNearTracker(time);
  const auto tracker = processor_->getListTracker().front();

  forceAssociatedUpdate(time, {14.5, 0.0});
  const int measurements_after_rejection = tracker->getTotalMeasurementCount();
  time += rclcpp::Duration(50ms);
  forceAssociatedUpdate(time, {10.2, 0.0});

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_after_rejection + 1);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 0);
  EXPECT_EQ(processor_->getListTracker().size(), 1U);
}

TEST_F(BirthGuardTest, NormalAssociatedUpdateRemainsAccepted)
{
  auto time = baseTime();
  establishNearTracker(time);
  const auto tracker = processor_->getListTracker().front();
  const int measurements_before = tracker->getTotalMeasurementCount();

  forceAssociatedUpdate(time, {10.4, 0.0});

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before + 1);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 0);
}

TEST_F(BirthGuardTest, DisabledUpdateGuardKeepsAssociatedUpdateBehavior)
{
  channels_.front().birth_guard.update_guard.enabled = false;
  resetProcessor();
  auto time = baseTime();
  establishNearTracker(time);
  const auto tracker = processor_->getListTracker().front();
  const int measurements_before = tracker->getTotalMeasurementCount();

  forceAssociatedUpdate(time, {14.5, 0.0});

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before + 1);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 0);
}

TEST_F(BirthGuardTest, VelocityBootstrapAcceptsMotionBeforeMinimumMeasurements)
{
  auto time = baseTime();
  process(time, {{10.0, 0.0}});
  ASSERT_EQ(processor_->getListTracker().size(), 1U);
  const auto tracker = processor_->getListTracker().front();
  ASSERT_EQ(tracker->getTotalMeasurementCount(), 1);

  time += rclcpp::Duration(50ms);
  forceAssociatedUpdate(time, {14.0, 0.0});

  EXPECT_EQ(tracker->getTotalMeasurementCount(), 2);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 0);
}

TEST_F(BirthGuardTest, LongCoastCannotGrowRadialAllowancePastCap)
{
  auto time = baseTime();
  establishNearTracker(time);
  const auto tracker = processor_->getListTracker().front();
  const int measurements_before = tracker->getTotalMeasurementCount();

  time += rclcpp::Duration(300ms);
  forceAssociatedUpdate(time, {14.75, 0.0});

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 1);
}

TEST_F(BirthGuardTest, MissingEgoUsesLooseEuclideanCap)
{
  auto time = baseTime();
  establishNearTracker(time);
  auto tracker = processor_->getListTracker().front();
  int measurements_before = tracker->getTotalMeasurementCount();

  forceAssociatedUpdate(time, {14.0, 0.0}, nullptr, false);

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before + 1);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 0);

  resetProcessor();
  time = baseTime();
  establishNearTracker(time);
  tracker = processor_->getListTracker().front();
  measurements_before = tracker->getTotalMeasurementCount();

  forceAssociatedUpdate(time, {15.1, 0.0}, nullptr, false);

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 1);
  EXPECT_EQ(processor_->getListTracker().size(), 1U);
}

TEST_F(BirthGuardTest, LooseEuclideanCapRejectsHugeJumpJustOutsideBearingGate)
{
  auto time = baseTime();
  establishNearTracker(time);
  const auto tracker = processor_->getListTracker().front();
  const int measurements_before = tracker->getTotalMeasurementCount();
  constexpr double angle_rad = 2.1 * 3.14159265358979323846 / 180.0;

  forceAssociatedUpdate(time, {20.0 * std::cos(angle_rad), 20.0 * std::sin(angle_rad)});

  EXPECT_EQ(tracker->getTotalMeasurementCount(), measurements_before);
  EXPECT_EQ(tracker->getNoMeasurementCount(), 1);
  EXPECT_EQ(processor_->getListTracker().size(), 1U);
}

TEST(BirthGuardConfigValidation, RejectsEveryNonFiniteDouble)
{
  using Config = mot::types::InputChannel::BirthGuard;
  constexpr std::array<double Config::*, 6> birth_double_fields = {
    &Config::hypothesis_timeout_sec,
    &Config::hypothesis_match_distance_m,
    &Config::hypothesis_max_speed_mps,
    &Config::conflict_max_coast_age_sec,
    &Config::conflict_min_range_gap_m,
    &Config::conflict_max_bearing_deg,
  };
  using UpdateConfig = Config::UpdateGuard;
  constexpr std::array<double UpdateConfig::*, 6> update_double_fields = {
    &UpdateConfig::base_allowance_m,
    &UpdateConfig::max_innovation_speed_mps,
    &UpdateConfig::max_elapsed_sec,
    &UpdateConfig::max_allowance_m,
    &UpdateConfig::max_bearing_deg,
    &UpdateConfig::max_euclidean_innovation_m,
  };

  EXPECT_TRUE(mot::types::isValidBirthGuardConfig(Config{}));
  for (const auto field : birth_double_fields) {
    for (const double invalid : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()}) {
      Config config;
      config.*field = invalid;
      EXPECT_FALSE(mot::types::isValidBirthGuardConfig(config));
    }
  }
  for (const auto field : update_double_fields) {
    for (const double invalid : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()}) {
      Config config;
      config.update_guard.*field = invalid;
      EXPECT_FALSE(mot::types::isValidBirthGuardConfig(config));
    }
  }

  Config allowance_order;
  allowance_order.update_guard.max_allowance_m =
    allowance_order.update_guard.base_allowance_m - 0.1;
  EXPECT_FALSE(mot::types::isValidBirthGuardConfig(allowance_order));

  Config insufficient_bootstrap;
  insufficient_bootstrap.update_guard.min_measurements = 2;
  EXPECT_FALSE(mot::types::isValidBirthGuardConfig(insufficient_bootstrap));

  Config euclidean_order;
  euclidean_order.update_guard.max_euclidean_innovation_m =
    euclidean_order.update_guard.max_allowance_m - 0.1;
  EXPECT_FALSE(mot::types::isValidBirthGuardConfig(euclidean_order));
}

TEST_F(BirthGuardTest, MissingEgoPoseWithholdsBirthUntilGeometryReturns)
{
  auto time = baseTime();
  for (int i = 0; i < 3; ++i) {
    process(time, {{30.0, 0.0}}, false);
    time += rclcpp::Duration(50ms);
  }
  ASSERT_TRUE(processor_->getListTracker().empty());

  process(time, {{30.0, 0.0}});

  EXPECT_EQ(processor_->getListTracker().size(), 1U);
}

TEST_F(BirthGuardTest, RejectsNonFiniteMeasurementPositions)
{
  const std::vector<std::pair<double, double>> invalid_positions = {
    {std::numeric_limits<double>::quiet_NaN(), 0.0},
    {0.0, std::numeric_limits<double>::infinity()},
    {-std::numeric_limits<double>::infinity(), 0.0},
  };

  auto time = baseTime();
  for (const auto & position : invalid_positions) {
    process(time, {position});
    time += rclcpp::Duration(50ms);
  }
  EXPECT_TRUE(processor_->getListTracker().empty());

  process(time, {{30.0, 0.0}});
  EXPECT_EQ(processor_->getListTracker().size(), 1U);
}

}  // namespace
