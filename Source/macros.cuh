/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

namespace ExposureRender
{

namespace Cuda
{

#define LAUNCH_DIMENSIONS(width, height, depth, block_width, block_height, block_depth)						\
																											\
	dim3 BlockDim;																							\
																											\
	BlockDim.x = block_width;																				\
	BlockDim.y = block_height;																				\
	BlockDim.z = block_depth;																				\
																											\
	dim3 GridDim;																							\
																											\
	GridDim.x = (int)ceilf((float)width / (float)BlockDim.x);												\
	GridDim.y = (int)ceilf((float)height / (float)BlockDim.y);												\
	GridDim.z = (int)ceilf((float)depth / (float)BlockDim.z);

#define LAUNCH_CUDA_KERNEL_TIMED(cudakernelcall, title)														\
{																											\
	cudaEvent_t EventStart, EventStop;																		\
																											\
	Cuda::HandleCudaError(cudaEventCreate(&EventStart));													\
	Cuda::HandleCudaError(cudaEventCreate(&EventStop));														\
	Cuda::HandleCudaError(cudaEventRecord(EventStart, 0));													\
																											\
	cudakernelcall;																							\
																											\
	Cuda::HandleCudaError(cudaGetLastError());																\
	Cuda::HandleCudaError(cudaThreadSynchronize());															\
																											\
	Cuda::HandleCudaError(cudaEventRecord(EventStop, 0));													\
	Cuda::HandleCudaError(cudaEventSynchronize(EventStop));													\
																											\
	float TimeDelta = 0.0f;																					\
																											\
	Cuda::HandleCudaError(cudaEventElapsedTime(&TimeDelta, EventStart, EventStop), title);					\
																											\
	/*gKernelTimings.Add(ErKernelTiming(title, TimeDelta));*/												\
																											\
	Cuda::HandleCudaError(cudaEventDestroy(EventStart));													\
	Cuda::HandleCudaError(cudaEventDestroy(EventStop));														\
}

#define LAUNCH_CUDA_KERNEL(cudakernelcall)																	\
{																											\
	cudakernelcall;																							\
																											\
	Cuda::HandleCudaError(cudaGetLastError());																\
	Cuda::HandleCudaError(cudaThreadSynchronize());															\
}

#define KERNEL_1D(width)																					\
	const int IDx 	= blockIdx.x * blockDim.x + threadIdx.x;												\
	const int IDt	= threadIdx.x;																			\
	const int IDk	= IDx;																					\
																											\
	if (IDx >= width)																						\
		return;

#define KERNEL_2D(width, height)																			\
	const int IDx 	= blockIdx.x * blockDim.x + threadIdx.x;												\
	const int IDy 	= blockIdx.y * blockDim.y + threadIdx.y;												\
	const int IDt	= threadIdx.y * blockDim.x + threadIdx.x;												\
	const int IDk	= IDy * width + IDx;																	\
																											\
	if (IDx >= width || IDy >= height)																		\
		return;

#define KERNEL_3D(width, height, depth)																		\
	const int IDx 	= blockIdx.x * blockDim.x + threadIdx.x;												\
	const int IDy 	= blockIdx.y * blockDim.y + threadIdx.y;												\
	const int IDz 	= blockIdx.z * blockDim.z + threadIdx.z;												\
	const int IDt	= threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;		\
	const int IDk	= IDz * width * height + IDy * width + IDx;												\
																											\
	if (IDx >= width || IDy >= height || IDz >= depth)														\
		return;

}

}
