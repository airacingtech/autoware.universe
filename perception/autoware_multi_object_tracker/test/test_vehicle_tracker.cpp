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

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/vehicle_tracker.hpp"
#include "autoware/multi_object_tracker/tracker/update/vehicle_update_strategy.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <rclcpp/time.hpp>

#include <autoware_perception_msgs/msg/shape.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace autoware::multi_object_tracker
{
namespace
{
// correctWheelAnchorLateral is pure scalar math. The measurement polygon is centered on the
// observed edge-center anchor (lateral_offset) with half-width polygon_half; the tracker is
// centered at 0 with half-width tracker_half.
//
// Aligning the fixed-width tracker box to each polygon edge gives two candidate vehicle centers;
// their segment is the lateral "dead-zone" (width |polygon_width - tracker_width|). The corrected
// lateral coordinate is the tracker center (0) projected into that dead-zone, and the added lateral
// variance is the dead-zone half-width squared.
struct LateralResult
{
  double lateral;  // corrected lateral coordinate (= lateral_offset + lateral_move)
  double var_lat;
};

LateralResult run(double tracker_width, double polygon_width, double lateral_offset)
{
  const double tracker_half = tracker_width * 0.5;
  const double polygon_half = polygon_width * 0.5;
  double var_lat = 0.0;
  const double lateral_move =
    correctWheelAnchorLateral(lateral_offset, tracker_half, polygon_half, var_lat);
  return {lateral_offset + lateral_move, var_lat};
}

// Variance is the dead-zone half-width squared, independent of the offset.
double expectedVar(double tracker_width, double polygon_width)
{
  const double half_dead_zone = 0.5 * std::abs(polygon_width - tracker_width);
  return half_dead_zone * half_dead_zone;
}

types::DynamicObject makeCenterObject(
  const double x, const double y, const double length = 4.5718, const double width = 1.8)
{
  types::DynamicObject object;
  object.channel_index = 0;
  object.existence_probability = 0.9F;
  object.classification = {{classes::Label::CAR, 1.0F}};
  object.pose.position.x = x;
  object.pose.position.y = y;
  object.pose.orientation.w = 1.0;
  object.pose_covariance.fill(0.0);
  object.pose_covariance[0] = 0.01;
  object.pose_covariance[7] = 0.01;
  object.pose_covariance[14] = 1.0;
  object.pose_covariance[21] = 1.0e6;
  object.pose_covariance[28] = 1.0e6;
  object.pose_covariance[35] = 1.0e6;
  object.kinematics.has_position_covariance = true;
  object.kinematics.orientation_availability = types::OrientationAvailability::UNAVAILABLE;
  object.kinematics.has_twist = false;
  object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  object.shape.dimensions.x = length;
  object.shape.dimensions.y = width;
  object.shape.dimensions.z = 1.2573;
  object.trust_extension = false;
  object.area = length * width;
  return object;
}
}  // namespace

TEST(VehicleCenterPosition, NominalLengthSurvivesTurningUpdatesAndPredictionGaps)
{
  constexpr double nominal_length = 4.5718;
  constexpr double radius = 30.0;
  constexpr double speed = 25.0;
  constexpr int64_t step_ns = 50000000;  // 20 Hz prediction

  const rclcpp::Time t0{0, 0, RCL_ROS_TIME};
  VehicleTracker tracker(
    object_model::normal_vehicle, t0, makeCenterObject(0.0, 0.0, nominal_length));

  types::InputChannel channel{};
  channel.index = 0;
  channel.trust_extension = false;
  channel.trust_orientation = false;
  channel.trust_position_as_center = true;

  // A 30 m-radius arc continually rotates the inferred heading. Measurements
  // arrive at 5 Hz while prediction runs at 20 Hz, exercising the dropout path
  // that previously let the two bicycle endpoints separate.
  for (int i = 1; i <= 800; ++i) {
    const rclcpp::Time time{static_cast<int64_t>(i) * step_ns, RCL_ROS_TIME};
    ASSERT_TRUE(tracker.predict(time));
    if (i == 1 || i % 4 == 0) {
      const double elapsed_s = static_cast<double>(i * step_ns) * 1.0e-9;
      const double angle = speed * elapsed_s / radius;
      auto measurement = makeCenterObject(
        radius * std::sin(angle), radius * (1.0 - std::cos(angle)),
        // Deliberately bogus alternating dimensions must remain untrusted.
        (i % 8 == 0) ? 20.0 : 1.0);
      ASSERT_TRUE(tracker.updateWithMeasurement(measurement, time, channel));
    }

    types::DynamicObject output;
    ASSERT_TRUE(tracker.getTrackedObject(time, output));
    EXPECT_TRUE(std::isfinite(output.shape.dimensions.x));
    EXPECT_NEAR(output.shape.dimensions.x, nominal_length, 1.0e-9) << "step=" << i;
  }
}

// Equal widths: zero dead-zone, the two candidate centers coincide -> anchor untouched, no
// variance.
TEST(CorrectWheelAnchorLateral, EqualWidthIsNoOp)
{
  const auto r = run(2.0, 2.0, 0.5);
  EXPECT_DOUBLE_EQ(r.lateral, 0.5);  // anchor unchanged
  EXPECT_DOUBLE_EQ(r.var_lat, 0.0);
}

// Row 1 - polygon smaller than the tracker and fully inside it (both overhangs negative): the
// tracker center sits inside the dead-zone, so the anchor snaps to the tracker lateral center.
TEST(CorrectWheelAnchorLateral, SmallPolygonWithinSnapsToCenter)
{
  const auto r = run(2.0, 1.0, 0.3);  // polygon [-0.2, 0.8] inside tracker [-1, 1]
  EXPECT_DOUBLE_EQ(r.lateral, 0.0);
  EXPECT_DOUBLE_EQ(r.var_lat, expectedVar(2.0, 1.0));  // 0.25
}

// Row 2 - polygon smaller than the tracker but one edge protrudes (one overhang positive): the
// protruding polygon edge is taken as a real vehicle edge and the box slides by that overhang.
TEST(CorrectWheelAnchorLateral, SmallPolygonProtrudingFollowsEdge)
{
  const auto r = run(2.0, 1.0, 0.8);  // polygon [0.3, 1.3], left edge protrudes past tracker 1.0
  EXPECT_DOUBLE_EQ(r.lateral, 0.3);   // tracker_center + overhang_left (1.3 - 1.0)
  EXPECT_DOUBLE_EQ(r.var_lat, expectedVar(2.0, 1.0));  // 0.25
}

// Row 3 - polygon larger than the tracker, tracker fully inside it (both overhangs positive): the
// tracker center is inside the dead-zone, so the anchor is held at the tracker center.
TEST(CorrectWheelAnchorLateral, WidePolygonStraddlingHoldsCenter)
{
  const auto r = run(2.0, 4.0, 0.5);  // polygon [-1.5, 2.5] straddles tracker [-1, 1]
  EXPECT_DOUBLE_EQ(r.lateral, 0.0);
  EXPECT_DOUBLE_EQ(r.var_lat, expectedVar(2.0, 4.0));  // 1.0
}

// Row 4 - polygon larger than the tracker but one edge recedes inside it (one overhang negative):
// the recessed polygon edge is the real vehicle edge; the anchor snaps to that edge-aligned center.
TEST(CorrectWheelAnchorLateral, WidePolygonRecedingFollowsEdge)
{
  const auto r = run(2.0, 4.0, 2.0);  // polygon [0, 4], right edge 0 recedes inside tracker
  EXPECT_DOUBLE_EQ(r.lateral, 1.0);   // tracker right edge aligned to polygon_right (0 + 1)
  EXPECT_DOUBLE_EQ(r.var_lat, expectedVar(2.0, 4.0));  // 1.0
}

// The projection is continuous as the tracker center crosses the dead-zone boundary.
TEST(CorrectWheelAnchorLateral, ContinuousAtBoundary)
{
  const double eps = 1e-6;
  const auto inside = run(2.0, 4.0, 1.0 - eps);
  const auto outside = run(2.0, 4.0, 1.0 + eps);
  EXPECT_NEAR(inside.lateral, outside.lateral, 1e-4);
  EXPECT_NEAR(inside.var_lat, outside.var_lat, 1e-4);
}

// Sign symmetry: a negative lateral offset mirrors the positive case.
TEST(CorrectWheelAnchorLateral, SignSymmetry)
{
  const auto pos = run(2.0, 4.0, 2.0);
  const auto neg = run(2.0, 4.0, -2.0);
  EXPECT_DOUBLE_EQ(pos.lateral, -neg.lateral);
  EXPECT_DOUBLE_EQ(pos.var_lat, neg.var_lat);
}

}  // namespace autoware::multi_object_tracker
