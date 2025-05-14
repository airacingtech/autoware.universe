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

#include "autoware/cuda_pointcloud_preprocessor/voxel_grid_kernels.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.
#include <cfloat>                     // For FLT_MAX

namespace autoware::cuda_pointcloud_preprocessor
{

// Internal structure to hold sum of coordinates and point count for a voxel
struct VoxelCentroidAccumulator
{
  float sum_x;
  float sum_y;
  float sum_z;
  int count;
};

// Internal structure to hold info about the point nearest to a voxel's centroid
struct VoxelNearestPointInfo
{
  float min_dist_sq;
  int point_original_idx; // Index in the original d_input_points array
};


// Helper device function to calculate voxel index
// Note: This is a simplified version. The PCL version has more robust calculations for min_b, div_b etc.
// These (min_b_x, inv_leaf_size_x, divb_mul_x etc.) would be calculated on CPU and passed to kernel,
// or calculated once in a setup kernel if bounds are dynamic.
__device__ int calculateVoxelIndex(
  float x, float y, float z,
  float min_x_bound, float min_y_bound, float min_z_bound,
  float inv_leaf_x, float inv_leaf_y, float inv_leaf_z,
  int grid_dim_x, int grid_dim_y, int /*grid_dim_z*/) // grid_dim_z not used in this 1D index formula directly
{
  int ijk0 = static_cast<int>(floorf(x * inv_leaf_x) - floorf(min_x_bound * inv_leaf_x));
  int ijk1 = static_cast<int>(floorf(y * inv_leaf_y) - floorf(min_y_bound * inv_leaf_y));
  int ijk2 = static_cast<int>(floorf(z * inv_leaf_z) - floorf(min_z_bound * inv_leaf_z));

  // Clamp indices to be within grid dimensions to prevent out-of-bounds access
  ijk0 = max(0, min(ijk0, grid_dim_x - 1));
  ijk1 = max(0, min(ijk1, grid_dim_y - 1));
  // ijk2 = max(0, min(ijk2, grid_dim_z - 1)); // Already clamped by access pattern if grid_dim_z is correct

  return ijk2 * (grid_dim_x * grid_dim_y) + ijk1 * grid_dim_x + ijk0;
}


// Kernel 1: Initialize temporary voxel data structures
__global__ void initializeVoxelDataKernel(
  VoxelCentroidAccumulator * voxel_centroids,
  VoxelNearestPointInfo * voxel_nearest_info,
  int num_total_voxels)
{
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_idx < num_total_voxels) {
    voxel_centroids[voxel_idx] = {0.0f, 0.0f, 0.0f, 0};
    voxel_nearest_info[voxel_idx] = {FLT_MAX, -1};
  }
}

// Kernel 2: Compute sums and counts for each voxel's centroid
__global__ void computePartialCentroidsKernel(
  const InputPointType * points,
  std::size_t num_points,
  VoxelCentroidAccumulator * voxel_centroids,
  float min_x_bound, float min_y_bound, float min_z_bound,
  float inv_leaf_x, float inv_leaf_y, float inv_leaf_z,
  int grid_dim_x, int grid_dim_y, int grid_dim_z,
  int num_total_voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx < num_points) {
    InputPointType p = points[point_idx];

    // Filter out invalid points (optional, if input can have NaNs)
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        return;
    }

    int voxel_id = calculateVoxelIndex(
      p.x, p.y, p.z,
      min_x_bound, min_y_bound, min_z_bound,
      inv_leaf_x, inv_leaf_y, inv_leaf_z,
      grid_dim_x, grid_dim_y, grid_dim_z);

    if (voxel_id >= 0 && voxel_id < num_total_voxels) { // Boundary check
      atomicAdd(&(voxel_centroids[voxel_id].sum_x), p.x);
      atomicAdd(&(voxel_centroids[voxel_id].sum_y), p.y);
      atomicAdd(&(voxel_centroids[voxel_id].sum_z), p.z);
      atomicAdd(&(voxel_centroids[voxel_id].count), 1);
    }
  }
}

// Kernel 3: Find the point nearest to each voxel's centroid
// This kernel iterates over points again. Each point computes its distance to its voxel's centroid
// and tries to become the "nearest point" for that voxel using atomic operations.
__global__ void findNearestPointsKernel(
  const InputPointType * points,
  std::size_t num_points,
  const VoxelCentroidAccumulator * voxel_centroids, // Read-only after Kernel 2
  VoxelNearestPointInfo * voxel_nearest_info,      // Updated atomically
  int min_points_per_voxel,
  float min_x_bound, float min_y_bound, float min_z_bound,
  float inv_leaf_x, float inv_leaf_y, float inv_leaf_z,
  int grid_dim_x, int grid_dim_y, int grid_dim_z,
  int num_total_voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx < num_points) {
    InputPointType p = points[point_idx];

    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        return;
    }

    int voxel_id = calculateVoxelIndex(
      p.x, p.y, p.z,
      min_x_bound, min_y_bound, min_z_bound,
      inv_leaf_x, inv_leaf_y, inv_leaf_z,
      grid_dim_x, grid_dim_y, grid_dim_z);

    if (voxel_id >= 0 && voxel_id < num_total_voxels) {
      VoxelCentroidAccumulator ca = voxel_centroids[voxel_id];
      if (ca.count >= min_points_per_voxel) {
        float centroid_x = ca.sum_x / ca.count;
        float centroid_y = ca.sum_y / ca.count;
        float centroid_z = ca.sum_z / ca.count;

        float dx = p.x - centroid_x;
        float dy = p.y - centroid_y;
        float dz = p.z - centroid_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        // Atomically update if this point is closer
        // This is a critical section. A simple atomicMin on dist_sq and then
        // storing point_idx can lead to races if two threads find a new minimum
        // simultaneously. A common way is to use atomicCAS in a loop, or if CUDA arch allows,
        // atomic operations on structs, or pack dist_sq and point_idx into a long long.
        // For simplicity, this uses a loop with atomicCAS on the distance.
        // A more robust solution might involve custom atomics or a lock-free approach.
        float old_dist = voxel_nearest_info[voxel_id].min_dist_sq;
        while (dist_sq < old_dist) {
            float previous_val = atomicCAS((float*)&(voxel_nearest_info[voxel_id].min_dist_sq), old_dist, dist_sq);
            if (previous_val == old_dist) { // Successfully updated min_dist_sq
                // Now, try to update the point_original_idx.
                // This is still not perfectly safe if another thread updates min_dist_sq
                // again before this thread updates point_original_idx.
                // For true atomicity of {dist, idx} pair, more advanced techniques are needed.
                // One such technique is to pack both into a single 64-bit integer if possible.
                voxel_nearest_info[voxel_id].point_original_idx = point_idx;
                break; // Exit loop
            }
            // If CAS failed, another thread updated old_dist. Retry with the new old_dist.
            old_dist = previous_val;
        }
      }
    }
  }
}

// Modified Kernel 4 to correctly access input points
__global__ void generateFinalMaskKernel( // Overload or rename, this is the corrected version
    std::size_t num_points,
    const VoxelNearestPointInfo * voxel_nearest_info,
    const VoxelCentroidAccumulator * voxel_centroids,
    int min_points_per_voxel,
    std::uint32_t * output_mask,
    float min_x_bound, float min_y_bound, float min_z_bound,
    float inv_leaf_x, float inv_leaf_y, float inv_leaf_z,
    int grid_dim_x, int grid_dim_y, int grid_dim_z,
    int num_total_voxels,
    const InputPointType * actual_input_points) // Added this parameter
  {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < num_points) {
      InputPointType p = actual_input_points[point_idx];
  
      if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
           output_mask[point_idx] = 0;
           return;
      }
  
      int voxel_id = calculateVoxelIndex(
        p.x, p.y, p.z,
        min_x_bound, min_y_bound, min_z_bound,
        inv_leaf_x, inv_leaf_y, inv_leaf_z,
        grid_dim_x, grid_dim_y, grid_dim_z);
  
      if (voxel_id >=0 && voxel_id < num_total_voxels && // Voxel ID is valid
          voxel_centroids[voxel_id].count >= min_points_per_voxel && // Voxel meets min points criteria
          voxel_nearest_info[voxel_id].point_original_idx == point_idx) { // Current point is the chosen one
        output_mask[point_idx] = 1;
      } else {
        output_mask[point_idx] = 0;
      }
    }
  }


// Host launch function
void voxelGridNearestCentroidLaunch(
  const InputPointType * d_input_points,
  std::size_t num_points,
  const VoxelGridParams & params,
  std::uint32_t * d_output_mask, // Assumed to be pre-allocated to num_points
  int threads_per_block,
  cudaStream_t & stream)
{
  if (num_points == 0) {
    return;
  }

  // Calculate grid dimensions from params
  // This mimics part of PCL's VoxelGrid::applyFilter setup
  float inv_leaf_x = 1.0f / params.leaf_size_x;
  float inv_leaf_y = 1.0f / params.leaf_size_y;
  float inv_leaf_z = 1.0f / params.leaf_size_z;

  // Integer bounds of the grid
  // These are conceptual; the calculateVoxelIndex handles direct mapping
  // int min_b_x = static_cast<int>(floorf(params.min_x_bound * inv_leaf_x));
  // int min_b_y = static_cast<int>(floorf(params.min_y_bound * inv_leaf_y));
  // int min_b_z = static_cast<int>(floorf(params.min_z_bound * inv_leaf_z));

  // int max_b_x = static_cast<int>(floorf(params.max_x_bound * inv_leaf_x));
  // int max_b_y = static_cast<int>(floorf(params.max_y_bound * inv_leaf_y));
  // int max_b_z = static_cast<int>(floorf(params.max_z_bound * inv_leaf_z));

  // Number of divisions (voxels) along each axis
  // Ensure positive dimensions, at least 1 voxel
  int grid_dim_x = static_cast<int>(ceilf((params.max_x_bound - params.min_x_bound) * inv_leaf_x));
  int grid_dim_y = static_cast<int>(ceilf((params.max_y_bound - params.min_y_bound) * inv_leaf_y));
  int grid_dim_z = static_cast<int>(ceilf((params.max_z_bound - params.min_z_bound) * inv_leaf_z));
  
  grid_dim_x = max(1, grid_dim_x);
  grid_dim_y = max(1, grid_dim_y);
  grid_dim_z = max(1, grid_dim_z);

  std::size_t num_total_voxels = static_cast<std::size_t>(grid_dim_x) * grid_dim_y * grid_dim_z;

  if (num_total_voxels == 0 || num_total_voxels > (1024 * 1024 * 256) /* Some reasonable upper limit */ ) {
    // Potentially too many voxels, handle error or default (e.g. pass all points)
    // For now, if grid is invalid, just clear the mask (or pass all points by setting mask to 1s)
    cudaMemsetAsync(d_output_mask, 0, num_points * sizeof(std::uint32_t), stream);
    // Or if you want to pass all points:
    // cudaMemsetAsync(d_output_mask, 1, num_points * sizeof(std::uint32_t), stream); // Requires int value for memset
    // For setting to 1, a simple kernel is better:
    // fillKernel<<< (num_points + threads_per_block - 1) / threads_per_block, threads_per_block, 0, stream >>>(d_output_mask, num_points, 1);
    return;
  }

  // Allocate temporary GPU memory
  VoxelCentroidAccumulator * d_voxel_centroids;
  VoxelNearestPointInfo * d_voxel_nearest_info;
  cudaMallocAsync(&d_voxel_centroids, num_total_voxels * sizeof(VoxelCentroidAccumulator), stream);
  cudaMallocAsync(&d_voxel_nearest_info, num_total_voxels * sizeof(VoxelNearestPointInfo), stream);

  // Kernel 1: Initialize
  int blocks_for_voxels = (num_total_voxels + threads_per_block - 1) / threads_per_block;
  initializeVoxelDataKernel<<<blocks_for_voxels, threads_per_block, 0, stream>>>(
    d_voxel_centroids, d_voxel_nearest_info, num_total_voxels);

  // Kernel 2: Compute partial centroids
  int blocks_for_points = (num_points + threads_per_block - 1) / threads_per_block;
  computePartialCentroidsKernel<<<blocks_for_points, threads_per_block, 0, stream>>>(
    d_input_points, num_points, d_voxel_centroids,
    params.min_x_bound, params.min_y_bound, params.min_z_bound,
    inv_leaf_x, inv_leaf_y, inv_leaf_z,
    grid_dim_x, grid_dim_y, grid_dim_z, num_total_voxels);

  // Kernel 3: Find nearest points to centroids
  findNearestPointsKernel<<<blocks_for_points, threads_per_block, 0, stream>>>(
    d_input_points, num_points, d_voxel_centroids, d_voxel_nearest_info,
    params.min_points_per_voxel,
    params.min_x_bound, params.min_y_bound, params.min_z_bound,
    inv_leaf_x, inv_leaf_y, inv_leaf_z,
    grid_dim_x, grid_dim_y, grid_dim_z, num_total_voxels);

  // Kernel 4: Generate final mask
  // Pass d_input_points to generateFinalMaskKernel to avoid the 0xBAD_POINTER/0xBEEF issue
 generateFinalMaskKernel<<<blocks_for_points, threads_per_block, 0, stream>>>(
    num_points, d_voxel_nearest_info, d_voxel_centroids,
    params.min_points_per_voxel,
    d_output_mask,
    params.min_x_bound, params.min_y_bound, params.min_z_bound,
    inv_leaf_x, inv_leaf_y, inv_leaf_z,
    grid_dim_x, grid_dim_y, grid_dim_z, num_total_voxels,
    d_input_points); // Pass actual input points

  // Free temporary GPU memory
  cudaFreeAsync(d_voxel_centroids, stream);
  cudaFreeAsync(d_voxel_nearest_info, stream);

  // Note: CUDA error checking (cudaGetLastError()) should be added after kernel launches
  // and memory operations for robust code.
}


}  // namespace autoware::cuda_pointcloud_preprocessor