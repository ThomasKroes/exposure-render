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

#include "Color.cuh"
#include "Filter.cuh"

#define KRNL_GAUSSIAN_FILTER_BLOCK_W		8
#define KRNL_GAUSSIAN_FILTER_BLOCK_H		8
#define KRNL_GAUSSIAN_FILTER_BLOCK_SIZE		KRNL_GAUSSIAN_FILTER_BLOCK_W * KRNL_GAUSSIAN_FILTER_BLOCK_H

KERNEL void KrnlGaussianFilterHorizontal(ColorXYZAf* pIn, ColorXYZAf* pOut, int Width, int Height)
{
	// Indexing
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;
	const int PID	= Y * Width + X;

	// Exit if beyond image boundaries
	if (X >= Width || Y >= Height)
		return;

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	Range[TID][0] = max((int)ceilf(X - gFrameEstimateFilter.KernelRadius), 0);
	Range[TID][1] = min((int)floorf(X + gFrameEstimateFilter.KernelRadius), Width - 1);

	// Filter accumulated sum color
	__shared__ ColorXYZAf Sum[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[TID].Set(0.0f, 0.0f, 0.0f, 0.0f);

	__shared__ float Weight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Weight[TID]			= 0.0f;
	TotalWeight[TID]	= 0.0f;

	// Execute kernel
	for (int x = Range[TID][0]; x <= Range[TID][1]; x++)
	{
		Weight[TID]			= gFrameEstimateFilter.KernelD[gFrameEstimateFilter.KernelRadius - (x - X)];
		Sum[TID]			+= pIn[Y * Width + x] * Weight[TID];
		TotalWeight[TID]	+= Weight[TID];
	}

	__syncthreads();

	if (TotalWeight[TID] > 0.0f)
		pOut[PID] = Sum[TID] / TotalWeight[TID];
	else
		pOut[PID] = ColorXYZAf(0.0f);
}

KERNEL void KrnlGaussianFilterVertical(ColorXYZAf* pIn, ColorXYZAf* pOut, int Width, int Height)
{
	// Indexing
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;
	const int PID	= Y * Width + X;

	// Exit if beyond image boundaries
	if (X >= Width || Y >= Height)
		return;

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	Range[TID][0] = max((int)ceilf(Y - gFrameEstimateFilter.KernelRadius), 0);
	Range[TID][1] = min((int)floorf(Y + gFrameEstimateFilter.KernelRadius), Height - 1);

	// Filter accumulated sum color
	__shared__ ColorXYZAf Sum[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[TID].Set(0.0f, 0.0f, 0.0f, 0.0f);

	__shared__ float Weight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Weight[TID]			= 0.0f;
	TotalWeight[TID]	= 0.0f;

	// Execute kernel
	for (int y = Range[TID][0]; y <= Range[TID][1]; y++)
	{
		Weight[TID]			= gFrameEstimateFilter.KernelD[gFrameEstimateFilter.KernelRadius - (y - Y)];
		Sum[TID]			+= pIn[y * Width + X] * Weight[TID];
		TotalWeight[TID]	+= Weight[TID];
	}

	__syncthreads();

	if (TotalWeight[TID] > 0.0f)
		pOut[PID] = Sum[TID] / TotalWeight[TID];
	else
		pOut[PID] = ColorXYZAf(0.0f);
}

void FilterGaussian(ColorXYZAf* pImage, ColorXYZAf* pTemp, int Width, int Height)
{
	const dim3 BlockDim(KRNL_GAUSSIAN_FILTER_BLOCK_W, KRNL_GAUSSIAN_FILTER_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilterHorizontal<<<GridDim, BlockDim>>>(pImage, pTemp, Width, Height)), "Gaussian filter (Horizontal)");
	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilterVertical<<<GridDim, BlockDim>>>(pTemp, pImage, Width, Height)), "Gaussian filter (Vertical)");
}