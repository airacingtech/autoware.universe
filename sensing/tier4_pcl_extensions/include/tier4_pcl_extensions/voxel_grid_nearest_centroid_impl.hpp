// Copyright 2020 Tier IV, Inc.
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
/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

 #ifndef TIER4_PCL_EXTENSIONS__VOXEL_GRID_NEAREST_CENTROID_IMPL_HPP_
 #define TIER4_PCL_EXTENSIONS__VOXEL_GRID_NEAREST_CENTROID_IMPL_HPP_
 
 #include "tier4_pcl_extensions/voxel_grid_nearest_centroid.hpp"
 
 #include <Eigen/Cholesky>
 #include <Eigen/Dense>
 
 #include <pcl/common/common.h>
 #include <pcl/pcl_config.h>
 
 #if PCL_VERSION < PCL_VERSION_CALC(1, 12, 0)
 #include <pcl/filters/boost.h>
 #endif
 
 #include <range/v3/all.hpp>
 
 #include <algorithm>
 #include <limits>
 #include <map>
 #include <vector>

//////////////////////////////////////////////////////////////////////////////////////////
#pragma omp declare reduction(merge_leaves : std::map<size_t, Leaf> : \
  omp_out = merge_leaves_maps(omp_out, omp_in)) \
  initializer(omp_priv = std::map<size_t, Leaf>())

// merge two leaf maps
std::map<size_t, Leaf> merge_leaves_maps(const std::map<size_t, Leaf>& map1, 
                                      const std::map<size_t, Leaf>& map2) {
  std::map<size_t, Leaf> result = map1;
  for (const auto& pair : map2) {
      auto it = result.find(pair.first);
      if (it == result.end()) {
          result[pair.first] = pair.second;
      } else {
          it->second.centroid += pair.second.centroid;
          it->second.nr_points += pair.second.nr_points;
      }
  }
  return result;
}

template <typename PointT>
void pcl::VoxelGridNearestCentroid<PointT>::applyFilter(PointCloud & output)
{
// Has the input dataset been set already?
if (!input_) {
  PCL_WARN("[pcl::%s::applyFilter] No input dataset given!\n", getClassName().c_str());
  output.width = output.height = 0;
  output.points.clear();
  return;
}

// Copy the header (and thus the frame_id) + allocate enough space for points
output.height = 1;       // downsampling breaks the organized structure
output.is_dense = true;  // we filter out invalid points
output.points.clear();

Eigen::Vector4f min_p, max_p;
// Get the minimum and maximum dimensions
if (!filter_field_name_.empty()) {  // If we don't want to process the entire cloud...
  getMinMax3D<PointT>(
    input_, filter_field_name_, static_cast<float>(filter_limit_min_),
    static_cast<float>(filter_limit_max_), min_p, max_p, filter_limit_negative_);
} else {
  getMinMax3D<PointT>(*input_, min_p, max_p);
}

// Check that the leaf size is not too small, given the size of the data
std::int64_t dx = static_cast<std::int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0]) + 1;
std::int64_t dy = static_cast<std::int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1]) + 1;
std::int64_t dz = static_cast<std::int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2]) + 1;

if ((dx * dy * dz) > std::numeric_limits<std::int32_t>::max()) {
  PCL_WARN(
    "[pcl::%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would "
    "overflow.",  // NOLINT
    getClassName().c_str());
  output.clear();
  return;
}

// Compute the minimum and maximum bounding box values
min_b_[0] = static_cast<int>(floor(min_p[0] * inverse_leaf_size_[0]));
max_b_[0] = static_cast<int>(floor(max_p[0] * inverse_leaf_size_[0]));
min_b_[1] = static_cast<int>(floor(min_p[1] * inverse_leaf_size_[1]));
max_b_[1] = static_cast<int>(floor(max_p[1] * inverse_leaf_size_[1]));
min_b_[2] = static_cast<int>(floor(min_p[2] * inverse_leaf_size_[2]));
max_b_[2] = static_cast<int>(floor(max_p[2] * inverse_leaf_size_[2]));

// Compute the number of divisions needed along all axis
div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
div_b_[3] = 0;

// Clear the leaves
leaves_.clear();

// cspell: ignore divb
// Set up the division multiplier
divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

int centroid_size = 4;

if (downsample_all_data_) {
  centroid_size = boost::mpl::size<FieldList>::value;
}

// ---[ RGB special case
std::vector<pcl::PCLPointField> fields;
int rgba_index = -1;
rgba_index = pcl::getFieldIndex<PointT>("rgb", fields);
if (rgba_index == -1) {
  rgba_index = pcl::getFieldIndex<PointT>("rgba", fields);
}
if (rgba_index >= 0) {
  rgba_index = fields[rgba_index].offset;
  centroid_size += 3;
}

// If we don't want to process the entire cloud, but rather filter points far
// away from the viewpoint first...
if (!filter_field_name_.empty()) {
  // Get the distance field index
  std::vector<pcl::PCLPointField> fields;
  int distance_idx = pcl::getFieldIndex<PointT>(filter_field_name_, fields);
  if (distance_idx == -1) {
    PCL_WARN(
      "[pcl::%s::applyFilter] Invalid filter field name. Index is %d.\n", getClassName().c_str(),
      distance_idx);
  }

  // First pass: Process points in parallel with reduction
  #pragma omp parallel reduction(merge_leaves : leaves_)
  {
      #pragma omp for nowait
      for (size_t cp = 0; cp < input_->points.size(); ++cp) {
          if (!input_->is_dense) {
              if (!std::isfinite(input_->points[cp].x) || 
                  !std::isfinite(input_->points[cp].y) ||
                  !std::isfinite(input_->points[cp].z)) {
                  continue;
              }
          }

          // Calculate voxel indices
          int ijk0 = static_cast<int>(
              floor(input_->points[cp].x * inverse_leaf_size_[0]) - static_cast<float>(min_b_[0]));
          int ijk1 = static_cast<int>(
              floor(input_->points[cp].y * inverse_leaf_size_[1]) - static_cast<float>(min_b_[1]));
          int ijk2 = static_cast<int>(
              floor(input_->points[cp].z * inverse_leaf_size_[2]) - static_cast<float>(min_b_[2]));

          int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

          Leaf & leaf = leaves_[idx];
          if (leaf.nr_points == 0) {
              leaf.centroid.resize(centroid_size);
              leaf.centroid.setZero();
          }

          // Process point data
          if (!downsample_all_data_) {
              Eigen::Vector4f pt(input_->points[cp].x, input_->points[cp].y, 
                               input_->points[cp].z, 0);
              leaf.centroid.template head<4>() += pt;
          } else {
              // Copy all the fields
              Eigen::VectorXf centroid = Eigen::VectorXf::Zero(centroid_size);
              // ---[ RGB special case
              if (rgba_index >= 0) {
                  // fill r/g/b data
                  int rgb;
                  memcpy(
                      &rgb, reinterpret_cast<const char *>(&input_->points[cp]) + rgba_index, sizeof(int));
                  centroid[centroid_size - 3] = static_cast<float>((rgb >> 16) & 0x0000ff);
                  centroid[centroid_size - 2] = static_cast<float>((rgb >> 8) & 0x0000ff);
                  centroid[centroid_size - 1] = static_cast<float>((rgb) & 0x0000ff);
              }
              pcl::for_each_type<FieldList>(
                  NdCopyPointEigenFunctor<PointT>(input_->points[cp], centroid));
              leaf.centroid += centroid;
          }
          ++leaf.nr_points;
      }
  }

} else {  // No distance filtering, process all data
  // First pass: Process points in parallel with reduction
  #pragma omp parallel reduction(merge_leaves : leaves_)
  {
      #pragma omp for nowait
      for (size_t cp = 0; cp < input_->points.size(); ++cp) {
          if (!input_->is_dense) {
              if (!std::isfinite(input_->points[cp].x) || 
                  !std::isfinite(input_->points[cp].y) ||
                  !std::isfinite(input_->points[cp].z)) {
                  continue;
              }
          }

          // Calculate voxel indices
          int ijk0 = static_cast<int>(
              floor(input_->points[cp].x * inverse_leaf_size_[0]) - static_cast<float>(min_b_[0]));
          int ijk1 = static_cast<int>(
              floor(input_->points[cp].y * inverse_leaf_size_[1]) - static_cast<float>(min_b_[1]));
          int ijk2 = static_cast<int>(
              floor(input_->points[cp].z * inverse_leaf_size_[2]) - static_cast<float>(min_b_[2]));

          int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

          Leaf & leaf = leaves_[idx];
          if (leaf.nr_points == 0) {
              leaf.centroid.resize(centroid_size);
              leaf.centroid.setZero();
          }

          // Process point data
          if (!downsample_all_data_) {
              Eigen::Vector4f pt(input_->points[cp].x, input_->points[cp].y, 
                               input_->points[cp].z, 0);
              leaf.centroid.template head<4>() += pt;
          } else {
              // Copy all the fields
              Eigen::VectorXf centroid = Eigen::VectorXf::Zero(centroid_size);
              // ---[ RGB special case
              if (rgba_index >= 0) {
                  // Fill r/g/b data, assuming that the order is BGRA
                  int rgb;
                  memcpy(
                      &rgb, reinterpret_cast<const char *>(&input_->points[cp]) + rgba_index, sizeof(int));
                  centroid[centroid_size - 3] = static_cast<float>((rgb >> 16) & 0x0000ff);
                  centroid[centroid_size - 2] = static_cast<float>((rgb >> 8) & 0x0000ff);
                  centroid[centroid_size - 1] = static_cast<float>((rgb) & 0x0000ff);
              }
              pcl::for_each_type<FieldList>(
                  NdCopyPointEigenFunctor<PointT>(input_->points[cp], centroid));
              leaf.centroid += centroid;
          }
          leaf.points.push_back(input_->points[cp]);
          ++leaf.nr_points;
      }
  }
}

// Second pass: Process leaves in parallel
std::vector<PointT> all_points;
std::vector<int> all_indices;
all_points.reserve(leaves_.size());
if (searchable_) {
  all_indices.reserve(leaves_.size());
}

// reduction for vectors
#pragma omp declare reduction(merge_vectors : std::vector<PointT> : \
    omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv = std::vector<PointT>())

#pragma omp declare reduction(merge_indices : std::vector<int> : \
    omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv = std::vector<int>())

#pragma omp parallel reduction(merge_vectors : all_points) reduction(merge_indices : all_indices)
{
    #pragma omp for nowait
    for (auto it = leaves_.begin(); it != leaves_.end(); ++it) {
        Leaf & leaf = it->second;
        leaf.centroid /= static_cast<float>(leaf.nr_points);

        if (leaf.nr_points >= min_points_per_voxel_) {
            // Find closest point to centroid
            const auto & centroid = leaf.centroid;
            float min_dist = std::numeric_limits<float>::max();
            size_t min_idx = 0;

            // Use a more efficient loop instead of ranges
            for (size_t i = 0; i < leaf.points.size(); ++i) {
                const auto& p = leaf.points[i];
                float dist = (p.x - centroid[0]) * (p.x - centroid[0]) +
                           (p.y - centroid[1]) * (p.y - centroid[1]) +
                           (p.z - centroid[2]) * (p.z - centroid[2]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = i;
                }
            }

            all_points.push_back(leaf.points[min_idx]);
            if (searchable_) {
                all_indices.push_back(static_cast<int>(it->first));
            }
        }
    }
}

// Copy results to output
output.points = std::move(all_points);
if (searchable_) {
  voxel_centroids_leaf_indices_ = std::move(all_indices);
}
output.width = static_cast<std::uint32_t>(output.points.size());
}

#define PCL_INSTANTIATE_VoxelGridNearestCentroid(T) \
template class PCL_EXPORTS pcl::VoxelGridNearestCentroid<T>;

#endif  // AUTOWARE__PCL_EXTENSIONS__VOXEL_GRID_NEAREST_CENTROID_IMPL_HPP_
