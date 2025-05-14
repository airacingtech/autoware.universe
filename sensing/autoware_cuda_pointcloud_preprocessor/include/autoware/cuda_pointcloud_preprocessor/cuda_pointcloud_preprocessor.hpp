// Copyright 2025 TIER IV, Inc.
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

#ifndef AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_POINTCLOUD_PREPROCESSOR_HPP_
#define AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_POINTCLOUD_PREPROCESSOR_HPP_

#include "autoware/cuda_pointcloud_preprocessor/point_types.hpp"
// #include "autoware/cuda_pointcloud_preprocessor/types.hpp" // Contains CropBoxParameters, RingOutlierFilterParameters, TwistStructs etc. Keep if any part of types.hpp is still needed.
                                                          // For a minimal voxel-grid only version, this might be mostly removable if those specific structs are not used.
                                                          // However, TransformStruct from types.hpp is likely used by common_kernels.hpp (transformPointsLaunch)
#include "autoware/cuda_pointcloud_preprocessor/voxel_grid_kernels.hpp" // For VoxelGridParams struct definition

#include <cuda_blackboard/cuda_pointcloud2.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
// The following can be removed if undistortion is not used:
// #include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
// #include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <thrust/device_vector.h>

#include <cstdint>
#include <deque> // Can be removed if twist_queue and angular_velocity_queue are removed from process()
#include <memory>
#include <vector>
#include <string> // For VoxelGridParams::output_frame_id

namespace autoware::cuda_pointcloud_preprocessor
{

// Forward declare from types.hpp if that entire include is removed but some structs are needed
// Or, ensure types.hpp is minimal and only contains what's necessary (like TransformStruct)
struct TransformStruct; // Assuming this is in types.hpp and used by transformPointsLaunch

class CudaPointcloudPreprocessor
{
public:
  // UndistortionType can be removed if undistortion is not used
  // enum class UndistortionType { Invalid, Undistortion2D, Undistortion3D };

  CudaPointcloudPreprocessor();
  ~CudaPointcloudPreprocessor(); // Provide a non-default destructor to handle CUDA resources

  // Remove setters for filters/features not being used
  // void setCropBoxParameters(const std::vector<CropBoxParameters> & crop_box_parameters);
  // void setRingOutlierFilterParameters(const RingOutlierFilterParameters & ring_outlier_parameters);
  // void setUndistortionType(const UndistortionType & undistortion_type);

  void setVoxelGridParameters(const VoxelGridParams & params); // Setter for Voxel Grid

  // preallocateOutput is internal, but if its logic changes, keep it.
  // void preallocateOutput(); // Made private if only called internally

  std::unique_ptr<cuda_blackboard::CudaPointCloud2> process(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg_ptr,
    const geometry_msgs::msg::TransformStamped & transform_msg,
    // The following queues can be removed if undistortion is not used by any remaining logic
    const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> & twist_queue,
    const std::deque<geometry_msgs::msg::Vector3Stamped> & angular_velocity_queue,
    const std::uint32_t first_point_rel_stamp_nsec
  );

private:
  void organizePointcloud();
  void preallocateOutput(); // Ensure this is called appropriately

  // Parameters for Voxel Grid (main filter)
  VoxelGridParams voxel_grid_parameters_;
  bool enable_voxel_grid_filter_; // To control if the filter runs

  // --- Removed parameters for other filters ---
  // CropBoxParameters self_crop_box_parameters_{};
  // CropBoxParameters mirror_crop_box_parameters_{};
  // RingOutlierFilterParameters ring_outlier_parameters_{};
  // UndistortionType undistortion_type_{UndistortionType::Invalid};

  // Basic point cloud properties
  int num_rings_{1}; // Default, updated by organizePointcloud
  int max_points_per_ring_{1}; // Default, updated by organizePointcloud
  std::size_t num_raw_points_{0};
  std::size_t num_organized_points_{0};

  std::vector<sensor_msgs::msg::PointField> point_fields_{};
  std::unique_ptr<cuda_blackboard::CudaPointCloud2> output_pointcloud_ptr_{};

  // CUDA specific members
  cudaStream_t stream_{nullptr};
  int max_blocks_per_grid_{0}; // Calculated based on SM count
  const int threads_per_block_{128}; // Default, can be configured
  cudaMemPool_t device_memory_pool_{nullptr};

  // Device memory buffers
  // Buffers for initial point organization
  thrust::device_vector<InputPointType> device_input_points_;
  thrust::device_vector<InputPointType> device_organized_points_;
  thrust::device_vector<std::int32_t> device_ring_index_;
  thrust::device_vector<std::uint32_t> device_indexes_tensor_;
  thrust::device_vector<std::uint32_t> device_sorted_indexes_tensor_;
  thrust::device_vector<std::int32_t> device_segment_offsets_;
  thrust::device_vector<std::int32_t> device_max_ring_;
  thrust::device_vector<std::int32_t> device_max_points_per_ring_;
  thrust::device_vector<std::uint8_t> device_sort_workspace_;
  std::size_t sort_workspace_bytes_{0};

  // Buffers for transformation and voxel grid filtering
  thrust::device_vector<InputPointType> device_transformed_points_;
  // Removed device_output_points_ as final output is directly to CudaPointCloud2::data
  thrust::device_vector<std::uint32_t> device_voxel_grid_mask_; // Dedicated mask for voxel grid
  thrust::device_vector<std::uint32_t> device_indices_; // For inclusive_scan output (indices for compaction)

  // --- Removed buffers for other filters/undistortion ---
  // thrust::device_vector<std::uint32_t> device_crop_mask_{};
  // thrust::device_vector<std::uint32_t> device_ring_outlier_mask_{};
  // thrust::device_vector<TwistStruct2D> device_twist_2d_structs_{};
  // thrust::device_vector<TwistStruct3D> device_twist_3d_structs_{};
  // thrust::device_vector<CropBoxParameters> host_crop_box_structs_{}; // This was a bug, should be std::vector
  // thrust::device_vector<CropBoxParameters> device_crop_box_structs_{};
};

}  // namespace autoware::cuda_pointcloud_preprocessor

#endif  // AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_POINTCLOUD_PREPROCESSOR_HPP_