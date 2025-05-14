#ifndef AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__VOXEL_GRID_KERNELS_HPP_
#define AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__VOXEL_GRID_KERNELS_HPP_

#include "autoware/cuda_pointcloud_preprocessor/point_types.hpp"
#include <cuda_runtime.h>

namespace autoware::cuda_pointcloud_preprocessor
{

// Parameters for the Voxel Grid Kernel
struct VoxelGridParams
{
  float leaf_size_x;
  float leaf_size_y;
  float leaf_size_z;

  // Min and max bounds of the point cloud to define the grid
  float min_x_bound;
  float min_y_bound;
  float min_z_bound;
  float max_x_bound; // Not strictly needed for grid calculation if using min_bound + dimensions
  float max_y_bound; // Not strictly needed
  float max_z_bound; // Not strictly needed

  // Derived internally in the launch function, but good to be aware of:
  // int min_b_x, min_b_y, min_b_z; // Minimum integer grid coordinates
  // int div_b_x, div_b_y, div_b_z; // Number of divisions (voxels) along each axis
  // int divb_mul_x, divb_mul_y, divb_mul_z; // Multipliers for 1D voxel index

  int min_points_per_voxel;
};

/**
 * @brief Launch function for the Voxel Grid Nearest Centroid filter.
 *
 * This function takes an input point cloud, applies a voxel grid filter where
 * for each voxel, the point closest to the voxel's centroid is kept.
 * The output is a mask array indicating which of the input points should be kept.
 *
 * @param input_points Pointer to the array of input points on the GPU.
 * @param num_points Total number of points in the input_points array.
 * @param params Parameters for the voxel grid filter (leaf sizes, cloud bounds).
 * @param output_mask Pointer to the output mask array on the GPU. Each element
 * corresponds to an input point. Set to 1 if the point is to be
 * kept, 0 otherwise. This array must be pre-allocated to num_points.
 * @param threads_per_block Number of threads per CUDA block.
 * @param stream CUDA stream for asynchronous execution.
 */
void voxelGridNearestCentroidLaunch(
  const InputPointType * d_input_points,
  std::size_t num_points,
  const VoxelGridParams & params,
  std::uint32_t * d_output_mask,
  int threads_per_block,
  cudaStream_t & stream);

}  // namespace autoware::cuda_pointcloud_preprocessor

#endif  // AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__VOXEL_GRID_KERNELS_HPP_