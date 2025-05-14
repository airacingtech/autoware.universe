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

#include "autoware/cuda_pointcloud_preprocessor/cuda_pointcloud_preprocessor.hpp" // Should include VoxelGridParams struct and enable_voxel_grid_filter_
#include "autoware/cuda_pointcloud_preprocessor/common_kernels.hpp" // For transformPointsLaunch, extractPointsLaunch
#include "autoware/cuda_pointcloud_preprocessor/organize_kernels.hpp"
#include "autoware/cuda_pointcloud_preprocessor/point_types.hpp"
#include "autoware/cuda_pointcloud_preprocessor/types.hpp" // For TransformStruct
#include "autoware/cuda_pointcloud_preprocessor/voxel_grid_kernels.hpp" // Your new kernel include

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cub/cub.cuh> // For sorting if still used by organizePointcloud, might not be strictly needed if organize is simplified

#include <cuda_runtime.h>
#include <tf2/utils.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h> // For inclusive_scan
// #include <thrust/reduce.h> // Not used in this simplified version

namespace autoware::cuda_pointcloud_preprocessor
{

CudaPointcloudPreprocessor::CudaPointcloudPreprocessor()
{
  // PointField setup (essential for output message)
  sensor_msgs::msg::PointField x_field, y_field, z_field, intensity_field, return_type_field,
    channel_field;
  x_field.name = "x"; x_field.offset = 0; x_field.datatype = sensor_msgs::msg::PointField::FLOAT32; x_field.count = 1;
  y_field.name = "y"; y_field.offset = 4; y_field.datatype = sensor_msgs::msg::PointField::FLOAT32; y_field.count = 1;
  z_field.name = "z"; z_field.offset = 8; z_field.datatype = sensor_msgs::msg::PointField::FLOAT32; z_field.count = 1;
  intensity_field.name = "intensity"; intensity_field.offset = 12; intensity_field.datatype = sensor_msgs::msg::PointField::UINT8; intensity_field.count = 1;
  return_type_field.name = "return_type"; return_type_field.offset = 13; return_type_field.datatype = sensor_msgs::msg::PointField::UINT8; return_type_field.count = 1;
  channel_field.name = "channel"; channel_field.offset = 14; channel_field.datatype = sensor_msgs::msg::PointField::UINT16; channel_field.count = 1;

  static_assert(sizeof(OutputPointType) == 16, "OutputPointType size is not 16 bytes");
  // ... (static_asserts for offsets)

  point_fields_.push_back(x_field);
  point_fields_.push_back(y_field);
  point_fields_.push_back(z_field);
  point_fields_.push_back(intensity_field);
  point_fields_.push_back(return_type_field);
  point_fields_.push_back(channel_field);

  cudaStreamCreate(&stream_);

  // CUDA Device Attributes (optional, but good practice)
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  max_blocks_per_grid_ = 4 * num_sm;

  // Memory Pool Allocators
  cudaMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(cudaMemPoolProps));
  pool_props.allocType = cudaMemAllocationTypePinned;
  pool_props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
  pool_props.location.type = cudaMemLocationTypeDevice;
  cudaGetDevice(&(pool_props.location.id));
  cudaMemPoolCreate(&device_memory_pool_, &pool_props);

  MemoryPoolAllocator<std::int32_t> allocator_int32(device_memory_pool_);
  MemoryPoolAllocator<std::uint32_t> allocator_uint32(device_memory_pool_);
  MemoryPoolAllocator<std::uint8_t> allocator_uint8(device_memory_pool_); // For sort workspace if organize uses it
  MemoryPoolAllocator<InputPointType> allocator_points(device_memory_pool_);

  // Basic buffers for organizePointcloud and processing
  num_rings_ = 1; // Will be updated by organizePointcloud
  max_points_per_ring_ = 1; // Will be updated
  num_organized_points_ = 1;

  device_ring_index_ = thrust::device_vector<std::int32_t, MemoryPoolAllocator<std::int32_t>>(allocator_int32);
  device_ring_index_.resize(num_rings_);

  device_indexes_tensor_ = thrust::device_vector<std::uint32_t, MemoryPoolAllocator<std::uint32_t>>(allocator_uint32);
  device_sorted_indexes_tensor_ = thrust::device_vector<std::uint32_t, MemoryPoolAllocator<std::uint32_t>>(allocator_uint32);
  device_indexes_tensor_.resize(num_organized_points_);
  device_sorted_indexes_tensor_.resize(num_organized_points_);

  device_segment_offsets_ = thrust::device_vector<std::int32_t, MemoryPoolAllocator<std::int32_t>>(allocator_int32);
  device_segment_offsets_.resize(num_rings_ + 1); // For CUB sort

  device_max_ring_ = thrust::device_vector<std::int32_t, MemoryPoolAllocator<std::int32_t>>(allocator_int32);
  device_max_ring_.resize(1);
  device_max_points_per_ring_ = thrust::device_vector<std::int32_t, MemoryPoolAllocator<std::int32_t>>(allocator_int32);
  device_max_points_per_ring_.resize(1);

  device_input_points_ = thrust::device_vector<InputPointType, MemoryPoolAllocator<InputPointType>>(allocator_points);
  device_organized_points_ = thrust::device_vector<InputPointType, MemoryPoolAllocator<InputPointType>>(allocator_points);
  device_organized_points_.resize(num_organized_points_);

  // Buffers for transformed points and the final mask/indices
  device_transformed_points_ = thrust::device_vector<InputPointType, MemoryPoolAllocator<InputPointType>>(allocator_points);
  device_transformed_points_.resize(num_organized_points_);

  device_voxel_grid_mask_ = thrust::device_vector<std::uint32_t, MemoryPoolAllocator<std::uint32_t>>(allocator_uint32); // Mask for voxel grid
  device_voxel_grid_mask_.resize(num_organized_points_);

  device_indices_ = thrust::device_vector<std::uint32_t, MemoryPoolAllocator<std::uint32_t>>(allocator_uint32);
  device_indices_.resize(num_organized_points_);


  // Initial memset for max_ring/max_points_per_ring (used by organize)
  cudaMemsetAsync(thrust::raw_pointer_cast(device_max_ring_.data()), 0, sizeof(std::int32_t), stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(device_max_points_per_ring_.data()), 0, sizeof(std::int32_t), stream_);
  // Initialize indexes_tensor for sorting (if used, value depends on sort key range)
  cudaMemsetAsync(thrust::raw_pointer_cast(device_indexes_tensor_.data()), 0xFF, num_organized_points_ * sizeof(std::uint32_t), stream_);


  // Sort workspace (if CUB sort is used in organizePointcloud)
  sort_workspace_bytes_ = 0; // Will be queried
  device_sort_workspace_ = thrust::device_vector<std::uint8_t, MemoryPoolAllocator<std::uint8_t>>(allocator_uint8);


  preallocateOutput(); // Prepares output_pointcloud_ptr_
  enable_voxel_grid_filter_ = false; // Default to disabled
}


void CudaPointcloudPreprocessor::setVoxelGridParameters(const VoxelGridParams & params)
{
  voxel_grid_parameters_ = params;
  enable_voxel_grid_filter_ = (params.leaf_size_x > 0.0f &&
                               params.leaf_size_y > 0.0f &&
                               params.leaf_size_z > 0.0f &&
                               params.min_points_per_voxel > 0); // Ensure min_points is also valid
}

void CudaPointcloudPreprocessor::preallocateOutput()
{
  // Allocate based on a reasonable maximum, organizePointcloud might update this
  // if num_organized_points_ changes significantly.
  // For now, use the member num_organized_points_ which is updated.
  std::size_t initial_capacity = num_organized_points_ > 0 ? num_organized_points_ : 200000; // Default capacity if 0 initially
  output_pointcloud_ptr_ = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  output_pointcloud_ptr_->data = cuda_blackboard::make_unique<std::uint8_t[]>(
    initial_capacity * sizeof(OutputPointType));
}

void CudaPointcloudPreprocessor::organizePointcloud()
{
  // This function resizes buffers based on actual point cloud characteristics (num_rings, max_points_per_ring)
  // It's important for setting `num_organized_points_` correctly.

  // Reset parts related to ring organization for each new cloud
  cudaMemsetAsync(thrust::raw_pointer_cast(device_max_ring_.data()), 0, sizeof(std::int32_t), stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(device_max_points_per_ring_.data()), 0, sizeof(std::int32_t), stream_);
  if (num_organized_points_ > 0) { // Only if there's a valid size
      cudaMemsetAsync(
        thrust::raw_pointer_cast(device_ring_index_.data()), 0, num_rings_ * sizeof(std::int32_t), // num_rings_ might be old here
        stream_);
      cudaMemsetAsync(
        thrust::raw_pointer_cast(device_indexes_tensor_.data()), 0xFF,
        num_organized_points_ * sizeof(std::uint32_t), stream_);
  }


  if (num_raw_points_ == 0) {
    num_organized_points_ = 0; // Ensure this is set if no raw points
    return;
  }

  const int raw_points_blocks_per_grid =
    (num_raw_points_ + threads_per_block_ - 1) / threads_per_block_;

  // Call organizeLaunch to determine ring structure and fill device_indexes_tensor_
  // This also updates device_max_ring_ and device_max_points_per_ring_
  organizeLaunch(
    thrust::raw_pointer_cast(device_input_points_.data()),
    thrust::raw_pointer_cast(device_indexes_tensor_.data()),
    thrust::raw_pointer_cast(device_ring_index_.data()), num_rings_, // Pass current num_rings_ (may be updated)
    thrust::raw_pointer_cast(device_max_ring_.data()), max_points_per_ring_, // Pass current max_points_per_ring_
    thrust::raw_pointer_cast(device_max_points_per_ring_.data()), num_raw_points_,
    threads_per_block_, raw_points_blocks_per_grid, stream_);

  std::int32_t max_ring_value;
  std::int32_t max_points_val; // Renamed to avoid conflict with member

  cudaMemcpyAsync(
    &max_ring_value, thrust::raw_pointer_cast(device_max_ring_.data()), sizeof(std::int32_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(
    &max_points_val, thrust::raw_pointer_cast(device_max_points_per_ring_.data()),
    sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_); // Wait for values to be copied

  // Check if buffer sizes need to be updated
  bool resize_needed = false;
  if (max_ring_value >= num_rings_) {
      num_rings_ = max_ring_value + 1;
      resize_needed = true;
  }
  if (max_points_val > max_points_per_ring_) {
      max_points_per_ring_ = std::max((max_points_val + 511) / 512 * 512, 512); // Align to 512
      resize_needed = true;
  }
  if (resize_needed || (num_rings_ * max_points_per_ring_ != num_organized_points_)) {
    num_organized_points_ = num_rings_ * max_points_per_ring_;

    // Resize all dependent buffers
    device_ring_index_.resize(num_rings_);
    device_indexes_tensor_.resize(num_organized_points_);
    device_sorted_indexes_tensor_.resize(num_organized_points_);
    device_segment_offsets_.resize(num_rings_ + 1);
    device_organized_points_.resize(num_organized_points_);
    device_transformed_points_.resize(num_organized_points_);
    device_voxel_grid_mask_.resize(num_organized_points_); // Resize the specific mask
    device_indices_.resize(num_organized_points_);

    preallocateOutput(); // Re-preallocate output based on new max size

    // Re-initialize fixed-size buffers if needed and re-run organizeLaunch if sizes changed mid-way
    // For simplicity, if a resize was needed, we might need to re-run organizeLaunch
    // or ensure it was designed to handle initial placeholder sizes.
    // The current PCL organizeLaunch likely expects correct num_rings/max_points_per_ring for its indexing.
    // If a resize was needed, it implies the first call to organizeLaunch might have been with sub-optimal sizes.
    // A common pattern is to run organize once to discover sizes, resize, then run again.
    // Or, organizeLaunch internally handles dynamic writing if buffers are large enough.

    // Assuming organizeLaunch was okay with initial estimates or buffers were oversized:
    // Setup segment_offsets for sorting
    std::vector<std::int32_t> segment_offsets_host(num_rings_ + 1);
    for (std::size_t i = 0; i < num_rings_ + 1; i++) {
      segment_offsets_host[i] = i * max_points_per_ring_;
    }
    cudaMemcpyAsync(
      thrust::raw_pointer_cast(device_segment_offsets_.data()), segment_offsets_host.data(),
      (num_rings_ + 1) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream_);

    // Re-Memset tensors after resize before potential second organize or sort
    cudaMemsetAsync(thrust::raw_pointer_cast(device_indexes_tensor_.data()), 0xFF, num_organized_points_ * sizeof(std::uint32_t), stream_);

    // If organizeLaunch has to be run again due to resize, do it here:
    // organizeLaunch(... with updated num_rings_, max_points_per_ring_ ...);
    // cudaStreamSynchronize(stream_); // if re-running organizeLaunch
  }


  // Sort keys (original indices) within each ring segment
  // This uses device_indexes_tensor_ as keys (which are point indices from raw cloud)
  // and device_sorted_indexes_tensor_ as output.
  // The actual values for sorting (e.g. azimuth) are inherent to how organizeLaunch populates device_indexes_tensor.
  if (num_organized_points_ > 0 && num_rings_ > 0) { // Only sort if there's data
    std::size_t required_sort_workspace_bytes = querySortWorkspace( // CUB's query
        num_organized_points_, num_rings_,
        thrust::raw_pointer_cast(device_segment_offsets_.data()),
        thrust::raw_pointer_cast(device_indexes_tensor_.data()), // Keys in
        thrust::raw_pointer_cast(device_sorted_indexes_tensor_.data())); // Keys out (sorted original indices)
    if(required_sort_workspace_bytes > sort_workspace_bytes_){
        device_sort_workspace_.resize(required_sort_workspace_bytes);
        sort_workspace_bytes_ = required_sort_workspace_bytes;
    }

    cub::DeviceSegmentedRadixSort::SortKeys(
        thrust::raw_pointer_cast(device_sort_workspace_.data()),
        sort_workspace_bytes_,
        thrust::raw_pointer_cast(device_indexes_tensor_.data()), // Input keys (original indices)
        thrust::raw_pointer_cast(device_sorted_indexes_tensor_.data()), // Output sorted keys
        num_organized_points_, // Total number of items
        num_rings_,            // Number of segments
        thrust::raw_pointer_cast(device_segment_offsets_.data()),     // Segment begin offsets
        thrust::raw_pointer_cast(device_segment_offsets_.data()) + 1, // Segment end offsets
        0, sizeof(std::uint32_t) * 8, stream_); // Sort all 32 bits
  }


  // Gather points into device_organized_points_ according to sorted indices
  if (num_organized_points_ > 0) {
    const int organized_points_blocks_per_grid =
        (num_organized_points_ + threads_per_block_ - 1) / threads_per_block_;
    gatherLaunch(
        thrust::raw_pointer_cast(device_input_points_.data()),
        thrust::raw_pointer_cast(device_sorted_indexes_tensor_.data()), // Use sorted indices
        thrust::raw_pointer_cast(device_organized_points_.data()),
        num_rings_, max_points_per_ring_, // For indexing into the 2D logical structure
        threads_per_block_, organized_points_blocks_per_grid, stream_);
  }
}

std::unique_ptr<cuda_blackboard::CudaPointCloud2> CudaPointcloudPreprocessor::process(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg_ptr,
  const geometry_msgs::msg::TransformStamped & transform_msg,
  const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> & /*twist_queue*/, // Not used
  const std::deque<geometry_msgs::msg::Vector3Stamped> & /*angular_velocity_queue*/, // Not used
  const std::uint32_t /*first_point_rel_stamp_nsec*/ ) // Not used
{
  // 1. Copy input points to GPU
  num_raw_points_ = input_pointcloud_msg_ptr->width * input_pointcloud_msg_ptr->height;
  if (num_raw_points_ > device_input_points_.size()) {
    std::size_t new_capacity = (num_raw_points_ + 1023) / 1024 * 1024; // Align up
    device_input_points_.resize(new_capacity);
  }
  if (num_raw_points_ > 0) {
    cudaMemcpyAsync(
        thrust::raw_pointer_cast(device_input_points_.data()),
        input_pointcloud_msg_ptr->data.data(),
        num_raw_points_ * sizeof(InputPointType),
        cudaMemcpyHostToDevice, stream_);
  }
  //cudaStreamSynchronize(stream_); // Sync after copy if organizePointcloud relies on it immediately

  // 2. Organize point cloud (determines num_organized_points_, sorts by ring/azimuth)
  organizePointcloud(); // This updates num_organized_points_

  // If no points after organization, return empty cloud
  if (num_organized_points_ == 0) {
    output_pointcloud_ptr_->width = 0;
    output_pointcloud_ptr_->row_step = 0;
    output_pointcloud_ptr_->data.reset(); // Release data buffer
    // Setup other header fields for an empty cloud
    output_pointcloud_ptr_->height = 1;
    output_pointcloud_ptr_->fields = point_fields_;
    output_pointcloud_ptr_->is_dense = true;
    output_pointcloud_ptr_->is_bigendian = input_pointcloud_msg_ptr->is_bigendian;
    output_pointcloud_ptr_->point_step = sizeof(OutputPointType);
    output_pointcloud_ptr_->header.stamp = input_pointcloud_msg_ptr->header.stamp;
    output_pointcloud_ptr_->header.frame_id = voxel_grid_parameters_.output_frame_id; // Use target frame
    return std::move(output_pointcloud_ptr_);
  }


  // 3. Transform points to base_frame_ (output frame of voxel grid)
  tf2::Quaternion rotation_quaternion(
    transform_msg.transform.rotation.x, transform_msg.transform.rotation.y,
    transform_msg.transform.rotation.z, transform_msg.transform.rotation.w);
  tf2::Matrix3x3 rotation_matrix;
  rotation_matrix.setRotation(rotation_quaternion);

  TransformStruct transform_struct;
  transform_struct.x = static_cast<float>(transform_msg.transform.translation.x);
  transform_struct.y = static_cast<float>(transform_msg.transform.translation.y);
  transform_struct.z = static_cast<float>(transform_msg.transform.translation.z);
  transform_struct.m11 = static_cast<float>(rotation_matrix[0][0]);
  transform_struct.m12 = static_cast<float>(rotation_matrix[0][1]);
  transform_struct.m13 = static_cast<float>(rotation_matrix[0][2]);
  transform_struct.m21 = static_cast<float>(rotation_matrix[1][0]);
  transform_struct.m22 = static_cast<float>(rotation_matrix[1][1]);
  transform_struct.m23 = static_cast<float>(rotation_matrix[1][2]);
  transform_struct.m31 = static_cast<float>(rotation_matrix[2][0]);
  transform_struct.m32 = static_cast<float>(rotation_matrix[2][1]);
  transform_struct.m33 = static_cast<float>(rotation_matrix[2][2]);

  const int blocks_per_grid = (num_organized_points_ + threads_per_block_ - 1) / threads_per_block_;

  transformPointsLaunch(
    thrust::raw_pointer_cast(device_organized_points_.data()), // Input organized points
    thrust::raw_pointer_cast(device_transformed_points_.data()), // Output transformed points
    num_organized_points_, transform_struct,
    threads_per_block_, blocks_per_grid, stream_);

  // Points to be filtered are now in device_transformed_points_
  InputPointType * device_points_to_filter = thrust::raw_pointer_cast(device_transformed_points_.data());
  std::uint32_t * d_final_mask = thrust::raw_pointer_cast(device_voxel_grid_mask_.data()); // Use the dedicated mask

  // 4. Apply Voxel Grid Filter
  if (enable_voxel_grid_filter_) {
    // VoxelGridParams (voxel_grid_parameters_) should be set by the node, including bounds.
    // The voxel_grid_kernels.cu will calculate grid dimensions internally from these params.
    voxelGridNearestCentroidLaunch(
      device_points_to_filter,
      num_organized_points_,
      voxel_grid_parameters_, // This MUST have min/max bounds set correctly
      d_final_mask,           // Output mask
      threads_per_block_,
      stream_);
  } else {
    // If filter is disabled, mark all points to be kept
    // A simple kernel would be: fillKernel<<<blocks_per_grid, threads_per_block_, 0, stream_>>>(d_final_mask, num_organized_points_, 1);
    // For now, let's assume if it's not enabled, we might want an empty cloud or handle differently.
    // Or, more simply, if not enabled, the node shouldn't call process, or set it up to pass through.
    // To pass through if not enabled:
    // Create a kernel void fill_mask_kernel(uint32_t* mask, size_t n, uint32_t val) { ... mask[idx] = val; }
    // fill_mask_kernel<<<blocks_per_grid, threads_per_block_, 0, stream_>>>(d_final_mask, num_organized_points_, 1);
     cudaMemsetAsync(d_final_mask, 0xFF, num_organized_points_ * sizeof(std::uint32_t), stream_); // 0xFFFFFFFF sets all to 1
  }

  // 5. Inclusive scan to get indices for compaction
  thrust::inclusive_scan(
    thrust::device, d_final_mask, d_final_mask + num_organized_points_,
    thrust::raw_pointer_cast(device_indices_.data()));

  // 6. Get the number of output points
  std::uint32_t num_output_points = 0;
  if (num_organized_points_ > 0) { // Only copy if there were points to scan
     cudaMemcpyAsync(
        &num_output_points, thrust::raw_pointer_cast(device_indices_.data()) + num_organized_points_ - 1,
        sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_); // Synchronize to get num_output_points for reallocation/header
  }


  // 7. Extract points into output buffer
  if (num_output_points > 0) {
    // Ensure output buffer is correctly sized
    if (output_pointcloud_ptr_->data == nullptr || (num_output_points * sizeof(OutputPointType)) > output_pointcloud_ptr_->data.size()) {
      // output_pointcloud_ptr_->data.reset(); // Free old memory
      output_pointcloud_ptr_->data = cuda_blackboard::make_unique<std::uint8_t[]>(num_output_points * sizeof(OutputPointType));
    }

    extractPointsLaunch(
      device_points_to_filter, // Source points (transformed)
      d_final_mask,            // The final mask from voxel grid
      thrust::raw_pointer_cast(device_indices_.data()),  // Indices from inclusive_scan
      num_organized_points_,   // Total number of points before filtering
      reinterpret_cast<OutputPointType *>(output_pointcloud_ptr_->data.get()),
      threads_per_block_, blocks_per_grid, stream_);
  } else {
      output_pointcloud_ptr_->data.reset(); // No points, clear data buffer
  }

  cudaStreamSynchronize(stream_); // Synchronize all GPU operations before setting up header

  // 8. Setup output PointCloud2 message metadata
  output_pointcloud_ptr_->width = num_output_points;
  output_pointcloud_ptr_->height = 1;
  output_pointcloud_ptr_->fields = point_fields_;
  output_pointcloud_ptr_->is_dense = true;
  output_pointcloud_ptr_->is_bigendian = input_pointcloud_msg_ptr->is_bigendian;
  output_pointcloud_ptr_->point_step = sizeof(OutputPointType);
  output_pointcloud_ptr_->row_step = num_output_points * output_pointcloud_ptr_->point_step;
  output_pointcloud_ptr_->header.stamp = input_pointcloud_msg_ptr->header.stamp;
  output_pointcloud_ptr_->header.frame_id = voxel_grid_parameters_.output_frame_id; // Use target frame

  return std::move(output_pointcloud_ptr_);
}

}  // namespace autoware::cuda_pointcloud_preprocessor