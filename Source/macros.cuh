/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
