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

#include "color.h"
#include "filter.h"
#include "tracer.h"

namespace ExposureRender
{

/*
#define KRNL_GAUSSIAN_FILTER_BLOCK_W		16
#define KRNL_GAUSSIAN_FILTER_BLOCK_H		8
#define KRNL_GAUSSIAN_FILTER_BLOCK_SIZE		KRNL_GAUSSIAN_FILTER_BLOCK_W * KRNL_GAUSSIAN_FILTER_BLOCK_H

KERNEL void KrnlGaussianFilterHorizontal(ColorXYZAf* pIn, ColorXYZAf* pOut)
{
	KERNEL_2D(gpFrameBuffer->Resolution[0], gpFrameBuffer->Resolution[1])

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	const GaussianFilter& Filter = gpTracer->FrameEstimateFilter;

	Range[IDt][0] = max((int)ceilf(IDx - Filter.KernelRadius), 0);
	Range[IDt][1] = min((int)floorf(IDx + Filter.KernelRadius), gpFrameBuffer->Resolution[0] - 1);

	// Filter accumulated sum color
	__shared__ ColorXYZAf Sum[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[IDt] = ColorXYZAf(0.0f, 0.0f, 0.0f, 0.0f);

	__shared__ float Weight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Weight[IDt]			= 0.0f;
	TotalWeight[IDt]	= 0.0f;

	// Execute kernel
	for (int x = Range[IDt][0]; x <= Range[IDt][1]; x++)
	{
		Weight[IDt]			= Filter.KernelD[Filter.KernelRadius - (x - IDx)];
		Sum[IDt]			+= pIn[IDy * gpFrameBuffer->Resolution[0] + x] * Weight[IDt];
		TotalWeight[IDt]	+= Weight[IDt];
	}

	__syncthreads();

	if (TotalWeight[IDt] > 0.0f)
		pOut[IDk] = Sum[IDt] / TotalWeight[IDt];
	else
		pOut[IDk] = ColorXYZAf(0.0f);
}

KERNEL void KrnlGaussianFilterVertical(ColorXYZAf* pIn, ColorXYZAf* pOut)
{
	KERNEL_2D(gpFrameBuffer->Resolution[0], gpFrameBuffer->Resolution[1])

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	const GaussianFilter& Filter = gpTracer->FrameEstimateFilter;

	Range[IDt][0] = max((int)ceilf(IDy - Filter.KernelRadius), 0);
	Range[IDt][1] = min((int)floorf(IDy + Filter.KernelRadius), gpFrameBuffer->Resolution[1] - 1);

	// Filter accumulated sum color
	__shared__ ColorXYZAf Sum[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[IDt] = ColorXYZAf(0.0f, 0.0f, 0.0f, 0.0f);

	__shared__ float Weight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Weight[IDt]			= 0.0f;
	TotalWeight[IDt]	= 0.0f;

	// Execute kernel
	for (int y = Range[IDt][0]; y <= Range[IDt][1]; y++)
	{
		Weight[IDt]			= Filter.KernelD[Filter.KernelRadius - (y - IDy)];
		Sum[IDt]			+= pIn[y * gpFrameBuffer->Resolution[0] + IDx] * Weight[IDt];
		TotalWeight[IDt]	+= Weight[IDt];
	}

	__syncthreads();

	if (TotalWeight[IDt] > 0.0f)
		pOut[IDk] = Sum[IDt] / TotalWeight[IDt];
	else
		pOut[IDk] = ColorXYZAf(0.0f);
}

void FilterGaussian(ColorXYZAf* pImage, ColorXYZAf* pTemp, int Width, int Height)
{
	LAUNCH_DIMENSIONS(Width, Height, 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilterHorizontal<<<GridDim, BlockDim>>>(pImage, pTemp)), "Gaussian filter (Horizontal)");
	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilterVertical<<<GridDim, BlockDim>>>(pTemp, pImage)), "Gaussian filter (Vertical)");
}
*/

}