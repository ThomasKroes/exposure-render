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

#include "Geometry.cuh"

#include <thrust/reduce.h>

namespace ExposureRender
{

#define KRNL_BENCHMARK_BLOCK_W		16 
#define KRNL_BENCHMARK_BLOCK_H		8
#define KRNL_BENCHMARK_BLOCK_SIZE	KRNL_BENCHMARK_BLOCK_W * KRNL_BENCHMARK_BLOCK_H

KERNEL void KrnlComputeNrmsError(FrameBuffer* pFrameBuffer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	ColorRGBAuc& RunningEstimate	= *pFrameBuffer->CudaDisplayEstimate.GetPtr(X, Y);
	ColorRGBAuc& ReferenceEstimate	= *pFrameBuffer->BenchmarkEstimateRgbaLdr.GetPtr(X, Y);

	const float ErrorRGB[3] = 
	{
		RunningEstimate.GetR() - ReferenceEstimate.GetR(),
		RunningEstimate.GetG() - ReferenceEstimate.GetG(),
		RunningEstimate.GetB() - ReferenceEstimate.GetB()
	};

	const float ErrorSquaredRGB[3] = 
	{
		ErrorRGB[0] * ErrorRGB[0],
		ErrorRGB[1] * ErrorRGB[1],
		ErrorRGB[2] * ErrorRGB[2]
	};

	const float NrmsError = (sqrtf(ErrorSquaredRGB[0]) + sqrtf(ErrorSquaredRGB[1]) + sqrtf(ErrorSquaredRGB[2])) / 3.0f;

	*pFrameBuffer->RmsError.GetPtr(X, Y) = NrmsError;
}

void ComputeAverageNrmsError(FrameBuffer& FB, FrameBuffer* pFrameBuffer, int Width, int Height, float& AverageNrmsError)
{
	const dim3 BlockDim(KRNL_BENCHMARK_BLOCK_W, KRNL_BENCHMARK_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	LAUNCH_CUDA_KERNEL_TIMED((KrnlComputeNrmsError<<<GridDim, BlockDim>>>(pFrameBuffer)), "Compute NRMS error");

	thrust::device_ptr<float> dev_ptr(FB.RmsError.GetPtr()); 

	float result = thrust::reduce(dev_ptr, dev_ptr + Width * Height);
	
	AverageNrmsError = (result / (float)(Width * Height)) / 255.0f;
}

}